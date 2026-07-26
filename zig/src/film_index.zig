const std = @import("std");
const sphtud = @import("sphtud");

const CHECKPOINT_EVERY: u64 = 10_000;
const CHECKPOINT_FILE = "film_index.txt";
const CHECKPOINT_TMP = "film_index.txt.tmp";
const MIN_TEXT_LEN: usize = 10_000;

const DecadeMap = std.AutoHashMap(u32, std.ArrayList(u64));

fn findDecade(text: []const u8) ?u32 {
    const prefix = "[[Category:";
    var search = text;
    while (std.mem.indexOf(u8, search, prefix)) |pos| {
        const after = search[pos + prefix.len ..];

        if (after.len < 4) break;

        var digits: usize = 0;
        while (digits < 4 and std.ascii.isDigit(after[digits])) : (digits += 1) {}

        if (digits == 4) {
            const year = std.fmt.parseInt(u32, after[0..4], 10) catch {
                search = search[pos + 1 ..];
                continue;
            };

            if (year >= 1880 and year <= 2050) {
                const rest = after[4..];
                // Match "YYYY films" or "YYYYs films"
                if (std.mem.startsWith(u8, rest, " films") or
                    std.mem.startsWith(u8, rest, "s films"))
                {
                    return year - (year % 10);
                }
            }
        }

        search = search[pos + 1 ..];
    }
    return null;
}

fn saveCheckpoint(decade_map: DecadeMap, file_pos: u64, pages: u64) !void {
    const fd = try sphtud.io.open(CHECKPOINT_TMP, .{
        .ACCMODE = .WRONLY,
        .CREAT = true,
        .TRUNC = true,
    }, 0o664);
    errdefer sphtud.io.close(fd);

    var write_buf: [4096]u8 = undefined;
    var w = sphtud.io.Writer.init(fd, &write_buf);

    try w.interface.print("pos {d}\n", .{file_pos});
    try w.interface.print("pages {d}\n", .{pages});

    // Sort decades for readable output
    var decade_keys: [32]u32 = undefined;
    var key_count: usize = 0;
    var kit = decade_map.keyIterator();
    while (kit.next()) |k| {
        if (key_count < decade_keys.len) {
            decade_keys[key_count] = k.*;
            key_count += 1;
        }
    }
    std.mem.sort(u32, decade_keys[0..key_count], {}, std.sort.asc(u32));

    for (decade_keys[0..key_count]) |decade| {
        const offsets = decade_map.get(decade) orelse continue;
        try w.interface.print("{d}", .{decade});
        for (offsets.items) |offset| {
            try w.interface.print(" {d}", .{offset});
        }
        try w.interface.writeAll("\n");
    }

    try w.interface.flush();
    sphtud.io.close(fd);

    const rc = sphtud.io.system.rename(CHECKPOINT_TMP, CHECKPOINT_FILE);
    switch (sphtud.io.system.errno(rc)) {
        .SUCCESS => {},
        else => return error.Rename,
    }
}

fn loadCheckpoint(decade_map: *DecadeMap, gpa: std.mem.Allocator, scratch: sphtud.alloc.LinearAllocator, pages_out: *u64) !?u64 {
    const cp = scratch.checkpoint();
    defer scratch.restore(cp);

    const fd = sphtud.io.open(CHECKPOINT_FILE, .{}, 0) catch return null;
    defer sphtud.io.close(fd);

    var read_buf: [4096]u8 = undefined;
    var r = sphtud.io.Reader.init(fd, &read_buf);
    const data = try r.interface.allocRemaining(scratch.allocator(), .unlimited);

    var resume_pos: u64 = 0;

    var lines = std.mem.tokenizeScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "pos ")) {
            resume_pos = std.fmt.parseInt(u64, line[4..], 10) catch continue;
        } else if (std.mem.startsWith(u8, line, "pages ")) {
            pages_out.* = std.fmt.parseInt(u64, line[6..], 10) catch continue;
        } else if (line.len > 0 and std.ascii.isDigit(line[0])) {
            var parts = std.mem.tokenizeScalar(u8, line, ' ');
            const decade_str = parts.next() orelse continue;
            const decade = std.fmt.parseInt(u32, decade_str, 10) catch continue;

            const gop = try decade_map.getOrPut(decade);
            if (!gop.found_existing) {
                gop.value_ptr.* = .empty;
            }

            while (parts.next()) |offset_str| {
                const offset = std.fmt.parseInt(u64, offset_str, 10) catch continue;
                try gop.value_ptr.append(gpa, offset);
            }
        }
    }

    return if (resume_pos > 0) resume_pos else null;
}

pub fn main(init: std.process.Init.Minimal) !void {
    var tpa: sphtud.alloc.TinyPageAllocator = undefined;
    try tpa.initPinned();

    var root_alloc: sphtud.alloc.Sphalloc = undefined;
    try root_alloc.initPinned(tpa.allocator(), "root");

    var scratch = sphtud.alloc.BufAllocator.init(try root_alloc.arena().alloc(u8, 1024 * 1024 * 10));

    const gpa = root_alloc.general();

    var decade_map = DecadeMap.init(gpa);
    var pages_processed: u64 = 0;

    const resume_pos: u64 = blk: {
        const p = try loadCheckpoint(&decade_map, gpa, scratch.linear(), &pages_processed);
        if (p) |pos| {
            std.debug.print("Resuming from file position {d}, pages already processed: {d}\n", .{ pos, pages_processed });
            break :blk pos;
        }
        break :blk 0;
    };

    var args = init.args.iterate();
    _ = args.next();
    const xml_path = args.next() orelse return error.NoXmlPath;

    const fd = try sphtud.io.open(xml_path, .{}, 0);
    defer sphtud.io.close(fd);

    if (resume_pos > 0) {
        _ = try sphtud.io.lseek(fd, @intCast(resume_pos), sphtud.io.system.SEEK.SET);
    }

    var read_buf: [4096]u8 = undefined;
    var file_reader = sphtud.io.Reader.init(fd, &read_buf);
    var xml = sphtud.xml.Parser.init(&file_reader.interface);

    const text_writer_buf = try root_alloc.arena().alloc(u8, 8 * 1024 * 1024);
    var text_writer = std.Io.Writer.fixed(text_writer_buf);
    var discard = std.Io.Writer.Discarding.init(&.{});

    var in_page = false;
    var in_text = false;
    var page_start: u64 = 0;

    while (true) {
        const w: *std.Io.Writer = if (in_text) &text_writer else &discard.writer;
        const item = xml.next(w) catch |e| switch (e) {
            error.EndOfStream => break,
            else => return e,
        } orelse break;

        switch (item.type) {
            .element_start => {
                if (std.mem.eql(u8, item.name, "page")) {
                    in_page = true;
                    page_start = resume_pos + item.stream_start;
                } else if (in_page and std.mem.eql(u8, item.name, "text")) {
                    text_writer = std.Io.Writer.fixed(text_writer_buf);
                    in_text = true;
                }
            },
            .element_end => {
                if (std.mem.eql(u8, item.name, "text") and in_text) {
                    in_text = false;
                    const text = text_writer.buffered();

                    const is_film = std.mem.indexOf(u8, text, "{{Infobox film") != null or
                        std.mem.indexOf(u8, text, "{{infobox film") != null;

                    if (is_film and text.len >= MIN_TEXT_LEN) {
                        if (findDecade(text)) |decade| {
                            const gop = try decade_map.getOrPut(decade);
                            if (!gop.found_existing) {
                                gop.value_ptr.* = .empty;
                            }
                            try gop.value_ptr.append(gpa, page_start);
                        }
                    }
                } else if (std.mem.eql(u8, item.name, "page") and in_page) {
                    in_page = false;
                    pages_processed += 1;

                    if (pages_processed % CHECKPOINT_EVERY == 0) {
                        const cur = resume_pos + item.stream_end;
                        std.debug.print("pages={d} pos={d}\n", .{ pages_processed, cur });
                        try saveCheckpoint(decade_map, cur, pages_processed);
                    }
                }
            },
            .element_content, .comment, .xml_decl => {},
        }
    }

    const final_pos = resume_pos + xml.stream_pos + xml.next_discard;
    try saveCheckpoint(decade_map, final_pos, pages_processed);

    std.debug.print("\nDone! pages processed: {d}\n", .{pages_processed});

    var kit = decade_map.keyIterator();
    var summary_keys: [32]u32 = undefined;
    var summary_count: usize = 0;
    while (kit.next()) |k| {
        if (summary_count < summary_keys.len) {
            summary_keys[summary_count] = k.*;
            summary_count += 1;
        }
    }
    std.mem.sort(u32, summary_keys[0..summary_count], {}, std.sort.asc(u32));

    for (summary_keys[0..summary_count]) |decade| {
        const offsets = decade_map.get(decade) orelse continue;
        std.debug.print("{d}s: {d} films\n", .{ decade, offsets.items.len });
    }
}
