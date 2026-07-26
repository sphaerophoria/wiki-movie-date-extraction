const std = @import("std");
const sphtud = @import("sphtud");
const wikipedia = @import("wikipedia.zig");

title: []const u8,
heading_cache: HeadingCache,
lex: sphtud.lex.Buf,
cur_section_paragraphs: ?std.mem.TokenIterator(u8, .sequence),
pending_header: ?[]const u8,
out_buf: []u8,
exhausted: bool,

const Preprocessor = @This();

pub fn init(alloc: std.mem.Allocator, title: []const u8, text: []const u8) !Preprocessor {
    const no_templates = try removeTemplates(alloc, text);
    var heading_cache = HeadingCache{
        .cache = try .initCapacity(alloc, 10),
    };
    try heading_cache.push("==Top==");
    return .{
        .out_buf = try alloc.alloc(u8, 256 * 1024),
        .title = title,
        .lex = sphtud.lex.Buf.init(no_templates),
        .cur_section_paragraphs = null,
        .pending_header = null,
        .exhausted = false,
        .heading_cache = heading_cache,
    };
}

pub fn next(self: *Preprocessor) !?[]const u8 {
    while (true) {
        if (self.cur_section_paragraphs) |*it| {
            while (it.next()) |p| {
                if (try self.formatParagraph(p)) |result| return result;
            }
            self.cur_section_paragraphs = null;
            if (self.pending_header) |h| {
                try self.heading_cache.push(h);
                self.pending_header = null;
            }
        }

        if (self.exhausted) return null;

        if (self.lex.takeUntilSequenceInclusive("==\n")) |range_in| {
            var range = range_in;
            range.end -= 1;
            const section = range.data(self.lex);

            const text_end_opt = std.mem.lastIndexOfScalar(u8, section, '\n');
            const header_start: ?usize = if (text_end_opt == null) 0 else if (text_end_opt.? < section.len - 1) text_end_opt.? + 1 else null;
            const text_end = if (text_end_opt) |e| e else 0;

            self.cur_section_paragraphs = std.mem.tokenizeSequence(u8, section[0..text_end], "\n\n");
            self.pending_header = if (header_start) |s| section[s..] else null;
        } else {
            self.cur_section_paragraphs = std.mem.tokenizeSequence(u8, self.lex.remaining(), "\n\n");
            self.exhausted = true;
        }
    }
}

fn formatParagraph(self: *Preprocessor, p: []const u8) !?[]const u8 {
    const p_trimmed = std.mem.trim(u8, p, &std.ascii.whitespace);
    if (p_trimmed.len == 0) return null;
    if (!containsMonthName(p_trimmed) or !containsYear(p_trimmed)) return null;

    var w = std.Io.Writer.fixed(self.out_buf);
    try w.writeAll("= ");
    try writeAsciiOnly(&w, self.title);
    try w.writeAll(" =\n");
    try self.heading_cache.format(&w);
    try writeAsciiOnly(&w, p_trimmed);
    try w.flush();
    return w.buffered();
}


const MONTH_NAMES = [_][]const u8{
    "january", "february", "march",     "april",   "may",      "june",
    "july",    "august",   "september", "october", "november", "december",
    "jan",     "feb",      "mar",       "apr",     "jun",      "jul",
    "aug",     "sep",      "oct",       "nov",     "dec",
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
                try writeAsciiOnly(w, item);
                try w.writeByte('\n');
            }
        }
    }
};

fn removeTemplates(alloc: std.mem.Allocator, data: []const u8) ![]const u8 {
    var alloc_writer = std.Io.Writer.Allocating.init(alloc);
    const w = &alloc_writer.writer;

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

fn writeAsciiOnly(w: *std.Io.Writer, data: []const u8) !void {
    var start: usize = 0;
    for (data, 0..) |c, i| {
        if (c > 127) {
            if (i > start) try w.writeAll(data[start..i]);
            start = i + 1;
        }
    }
    if (start < data.len) try w.writeAll(data[start..]);
}

