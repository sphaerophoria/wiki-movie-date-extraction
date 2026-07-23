const std = @import("std");
const sphtud = @import("sphtud");
const wikipedia = @import("wikipedia.zig");

const INDEX_PATH = "film_index.txt";
const SAMPLES_PER_DECADE = 400;
const DECADES = [_]u32{ 1990, 2000, 2010, 2020 };

const MONTH_NAMES = [_][]const u8{
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
};

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    for (0..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

fn containsMonthName(text: []const u8) bool {
    for (MONTH_NAMES) |month| {
        if (containsIgnoreCase(text, month)) return true;
    }
    return false;
}

fn containsYear(text: []const u8) bool {
    var i: usize = 0;
    while (i + 4 <= text.len) : (i += 1) {
        if (!std.ascii.isDigit(text[i])) continue;
        if (!std.ascii.isDigit(text[i + 1])) continue;
        if (!std.ascii.isDigit(text[i + 2])) continue;
        if (!std.ascii.isDigit(text[i + 3])) continue;
        if (i > 0 and std.ascii.isDigit(text[i - 1])) continue;
        if (i + 4 < text.len and std.ascii.isDigit(text[i + 4])) continue;
        const year = std.fmt.parseInt(u32, text[i .. i + 4], 10) catch continue;
        if (year >= 1800 and year <= 2200) return true;
    }
    return false;
}

const HeadingCache = struct {
    cache: std.ArrayList([]const u8),

    pub fn push(self: *HeadingCache, s: []const u8) !void {
        const level = level: {
            for (s, 0..) |c, i| {
                if (c != '=') break :level i;
            }
            break :level s.len;
        };

        for (@min(self.cache.items.len, level)..level) |_| {
            try self.cache.appendBounded("");
        }

        while (self.cache.items.len > level) {
            _ = self.cache.pop();
        }

        try self.cache.appendBounded(s);
    }

    pub fn format(self: HeadingCache, w: *std.Io.Writer) !void {
        for (self.cache.items) |item| {
            if (item.len > 0) {
                try w.print("{s}\n", .{item});
            }
        }
    }
};

fn removeTemplates(buf: []u8, data: []const u8) ![]const u8 {
    var w = std.Io.Writer.fixed(buf);

    var lex = sphtud.lex.Buf.init(data);
    while (lex.takeUntilSequence("{{")) |range| {
        var consumer = try wikipedia.TemplateIterator.initLex(lex);
        defer _ = lex.commit(consumer.buf);

        while (try consumer.next()) |_| {}
        try w.print("{s}\n", .{range.data(lex)});
    }

    try w.flush();
    return w.buffered();
}

fn writeParagraph(heading_cache: *HeadingCache, outdir: c_int, p: []const u8, file_name: []const u8, segment_idx: usize) !void {
    const p_trimmed = std.mem.trim(u8, p, &std.ascii.whitespace);
    if (p_trimmed.len == 0) return;
    if (!containsMonthName(p_trimmed) or !containsYear(p_trimmed)) return;

    var segment_buf: [256]u8 = undefined;
    const segment_name = try std.fmt.bufPrintZ(&segment_buf, "{s}_{d}", .{ file_name, segment_idx });

    const outf = try sphtud.io.openat(outdir, segment_name, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o664);
    defer sphtud.io.close(outf);

    var writer_buf: [4096]u8 = undefined;
    var outw = sphtud.io.Writer.init(outf, &writer_buf);

    try outw.interface.print("= {s} =\n", .{file_name});
    try heading_cache.format(&outw.interface);
    try outw.interface.writeAll(p_trimmed);
    try outw.interface.flush();
}

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
            if (d == decade) { valid = true; break; }
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

    const preprocess_buf = try scratch.allocator().alloc(u8, 8 * 1024 * 1024);
    const preprocessed = try removeTemplates(preprocess_buf, text);

    var cache_buf: [10][]const u8 = undefined;
    var heading_cache = HeadingCache{ .cache = .initBuffer(&cache_buf) };
    try heading_cache.push("==Top==");

    var lex = sphtud.lex.Buf.init(preprocessed);
    var segment_idx: usize = 0;

    while (lex.takeUntilSequenceInclusive("==\n")) |range_in| {
        var range = range_in;
        range.end -= 1;
        const section = range.data(lex);

        const text_end_opt = std.mem.lastIndexOfScalar(u8, section, '\n');
        const header_start: ?usize = if (text_end_opt == null) 0
            else if (text_end_opt.? < section.len - 1) text_end_opt.? + 1
            else null;
        const text_end = if (text_end_opt) |e| e else 0;

        var paragraph_it = std.mem.tokenizeSequence(u8, section[0..text_end], "\n\n");
        while (paragraph_it.next()) |p| {
            try writeParagraph(&heading_cache, outdir, p, title, segment_idx);
            segment_idx += 1;
        }

        if (header_start) |s| {
            try heading_cache.push(section[s..]);
        }
    }

    var paragraph_it = std.mem.tokenizeSequence(u8, lex.remaining(), "\n\n");
    while (paragraph_it.next()) |p| {
        try writeParagraph(&heading_cache, outdir, p, title, segment_idx);
        segment_idx += 1;
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
