const std = @import("std");
const sphtud = @import("sphtud");
const Preprocessor = @import("Preprocessor.zig");

const INDEX_PATH = "film_index.txt";
const SAMPLES_PER_DECADE = 400;
const DECADES = [_]u32{ 1990, 2000, 2010, 2020 };

fn loadIndex(alloc: std.mem.Allocator, scratch: sphtud.alloc.LinearAllocator) !std.AutoHashMap(u32, std.ArrayList(u64)) {
    const cp = scratch.checkpoint();
    defer scratch.restore(cp);

    var ret = std.AutoHashMap(u32, std.ArrayList(u64)).init(alloc);

    const fd = try sphtud.io.open(INDEX_PATH, .{}, 0);
    defer sphtud.io.close(fd);

    var read_buf: [4096]u8 = undefined;
    var r = sphtud.io.Reader.init(fd, &read_buf);
    const data = try r.interface.allocRemaining(scratch.allocator(), .unlimited);

    var lines = std.mem.tokenizeScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or !std.ascii.isDigit(line[0])) continue;

        var parts = std.mem.tokenizeScalar(u8, line, ' ');
        const decade_str = parts.next() orelse continue;
        const decade = std.fmt.parseInt(u32, decade_str, 10) catch continue;

        var valid = false;
        for (DECADES) |d| {
            if (d == decade) {
                valid = true;
                break;
            }
        }
        if (!valid) continue;

        const gop = try ret.getOrPut(decade);
        if (!gop.found_existing) gop.value_ptr.* = .empty;

        while (parts.next()) |offset_str| {
            const offset = std.fmt.parseInt(u64, offset_str, 10) catch continue;
            try gop.value_ptr.append(alloc, offset);
        }
    }

    return ret;
}

fn sanitizeTitle(raw: []const u8, buf: []u8) []u8 {
    var len: usize = 0;
    for (raw) |c| {
        if (len >= buf.len) break;
        buf[len] = switch (c) {
            ' ' => '_',
            '/' => '-',
            else => c,
        };
        len += 1;
    }
    return buf[0..len];
}

fn processWikitext(scratch: sphtud.alloc.LinearAllocator, title: []const u8, text: []const u8, outdir: c_int) !void {
    const cp = scratch.checkpoint();
    defer scratch.restore(cp);

    var pp = try Preprocessor.init(scratch.allocator(), title, text);

    var segment_idx: usize = 0;
    while (try pp.next()) |content| {
        var segment_buf: [256]u8 = undefined;
        const segment_name = try std.fmt.bufPrintZ(&segment_buf, "{s}_{d}", .{ title, segment_idx });
        segment_idx += 1;

        const outf = try sphtud.io.openat(outdir, segment_name, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o664);
        defer sphtud.io.close(outf);

        var writer_buf: [4096]u8 = undefined;
        var outw = sphtud.io.Writer.init(outf, &writer_buf);
        try outw.interface.writeAll(content);
        try outw.interface.flush();
    }

    std.debug.print("  processed {s}\n", .{title});
}

fn processRawFile(scratch: sphtud.alloc.LinearAllocator, file_path: [:0]const u8, outdir: c_int) !void {
    const cp = scratch.checkpoint();
    defer scratch.restore(cp);

    const basename = std.fs.path.basename(file_path);
    var title_buf: [512]u8 = undefined;
    const title = sanitizeTitle(basename, &title_buf);

    const fd = try sphtud.io.open(file_path, .{}, 0);
    defer sphtud.io.close(fd);

    var read_buf: [4096]u8 = undefined;
    var reader = sphtud.io.Reader.init(fd, &read_buf);
    const text = try reader.interface.allocRemaining(scratch.allocator(), .unlimited);

    try processWikitext(scratch, title, text, outdir);
}

fn processPage(scratch: sphtud.alloc.LinearAllocator, reader: *sphtud.io.Reader, offset: u64, outdir: c_int) !void {
    const cp = scratch.checkpoint();
    defer scratch.restore(cp);

    const text_buf = try scratch.allocator().alloc(u8, 8 * 1024 * 1024);

    try reader.seekTo(@intCast(offset));

    var xml = sphtud.xml.Parser.init(&reader.interface);
    var text_writer = std.Io.Writer.fixed(text_buf);
    var discard = std.Io.Writer.Discarding.init(&.{});

    var title_buf: [512]u8 = undefined;
    var title_len: usize = 0;
    var in_page = false;
    var in_title = false;
    var in_text = false;

    while (true) {
        const w: *std.Io.Writer = if (in_title or in_text) &text_writer else &discard.writer;
        const item = xml.next(w) catch |e| switch (e) {
            error.EndOfStream => return,
            else => return e,
        } orelse return;

        switch (item.type) {
            .element_start => {
                if (std.mem.eql(u8, item.name, "page")) {
                    in_page = true;
                } else if (in_page and std.mem.eql(u8, item.name, "title")) {
                    text_writer = std.Io.Writer.fixed(text_buf);
                    in_title = true;
                } else if (in_page and std.mem.eql(u8, item.name, "text")) {
                    text_writer = std.Io.Writer.fixed(text_buf);
                    in_text = true;
                }
            },
            .element_end => {
                if (std.mem.eql(u8, item.name, "title") and in_title) {
                    in_title = false;
                    const sanitized = sanitizeTitle(text_writer.buffered(), &title_buf);
                    title_len = sanitized.len;
                } else if (std.mem.eql(u8, item.name, "text") and in_text) {
                    in_text = false;
                    const text = text_writer.buffered();
                    if (title_len == 0 or text.len == 0) return;

                    const title = title_buf[0..title_len];
                    try processWikitext(scratch, title, text, outdir);
                    return;
                } else if (std.mem.eql(u8, item.name, "page") and in_page) {
                    return;
                }
            },
            .element_content, .comment, .xml_decl => {},
        }
    }
}

pub fn main(init: std.process.Init.Minimal) !void {
    var tpa: sphtud.alloc.TinyPageAllocator = undefined;
    try tpa.initPinned();

    var root_alloc: sphtud.alloc.Sphalloc = undefined;
    try root_alloc.initPinned(tpa.allocator(), "root");

    var scratch = sphtud.alloc.BufAllocator.init(try root_alloc.arena().alloc(u8, 1024 * 1024 * 20));
    const gpa = root_alloc.general();

    const args = try init.args.toSlice(root_alloc.arena());

    if (args.len >= 2 and std.mem.eql(u8, args[1], "pages")) {
        if (args.len < 4) {
            std.debug.print("Usage: preprocess pages <output-dir> <file>...\n", .{});
            return error.MissingArgument;
        }
        const outdir_path = args[2];
        const outdir = try sphtud.io.open(outdir_path, .{ .DIRECTORY = true }, 0);
        defer sphtud.io.close(outdir);

        for (args[3..]) |file_path| {
            processRawFile(scratch.linear(), file_path, outdir) catch |e| {
                std.debug.print("  error processing {s}: {}\n", .{ file_path, e });
            };
        }
        std.debug.print("Done!\n", .{});
        return;
    }

    if (args.len < 3) {
        std.debug.print("Usage: preprocess <input_file> <output-dir>\n", .{});
        std.debug.print("       preprocess pages <output-dir> <file>...\n", .{});
        return error.MissingArgument;
    }
    const xml_path = args[1];
    const outdir_path = args[2];

    std.debug.print("Loading index...\n", .{});
    const decade_map = try loadIndex(gpa, scratch.linear());

    const xml_fd = try sphtud.io.open(xml_path, .{}, 0);
    defer sphtud.io.close(xml_fd);

    const outdir = try sphtud.io.open(outdir_path, .{ .DIRECTORY = true }, 0);
    defer sphtud.io.close(outdir);

    var read_buf: [4096]u8 = undefined;
    var reader = sphtud.io.Reader.init(xml_fd, &read_buf);

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    for (DECADES) |decade| {
        const cp = scratch.checkpoint();
        defer scratch.restore(cp);

        const offsets_list = decade_map.get(decade) orelse {
            std.debug.print("{d}s: no entries\n", .{decade});
            continue;
        };
        const offsets = offsets_list.items;
        std.debug.print("{d}s: {d} films, sampling {d}\n", .{ decade, offsets.len, @min(SAMPLES_PER_DECADE, offsets.len) });

        const indices = try scratch.allocator().alloc(usize, offsets.len);
        for (indices, 0..) |*idx, i| idx.* = i;

        const take = @min(SAMPLES_PER_DECADE, offsets.len);
        for (0..take) |i| {
            const j = rand.intRangeLessThan(usize, i, offsets.len);
            const tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        for (indices[0..take]) |idx| {
            processPage(scratch.linear(), &reader, offsets[idx], outdir) catch |e| {
                std.debug.print("  error at offset {d}: {}\n", .{ offsets[idx], e });
            };
        }
    }

    std.debug.print("Done!\n", .{});
}
