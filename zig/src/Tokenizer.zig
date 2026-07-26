const std = @import("std");
const sphtud = @import("sphtud");

const StringToIdx = sphtud.util.StringHashMap(i64);
string_to_idx: StringToIdx,
vocab: [][]const u8,

const Tokenizer = @This();

const minilm_vocab = @embedFile("res/tokenizer/minilm_vocab.txt");

pub fn init(arena: std.mem.Allocator) !Tokenizer {
    var r = std.Io.Reader.fixed(minilm_vocab);

    const typical = 32 * 1024;
    // our vocab was ~30k. 16x seems like good headroom
    const max = typical * 16;

    var ret = try sphtud.util.StringHashMap(i64).init(
        arena,
        .linear(arena),
        typical,
        max,
    );

    var vocab = std.ArrayList([]const u8).empty;
    var idx: usize = 0;
    while (try r.takeDelimiter('\n')) |line| {
        const key = try arena.dupe(u8, line);
        try ret.put(key, @intCast(idx));
        try vocab.append(arena, key);
        idx += 1;
    }

    return .{
        .string_to_idx = ret,
        .vocab = vocab.items,
    };
}

pub const TokenOffset = struct {
    start: u32,
    end: u32,
};

pub const TokenizeResult = struct {
    tokens: []i64,
    offsets: []TokenOffset,
};

pub fn tokenize(self: Tokenizer, alloc: std.mem.Allocator, input_text: []const u8) !TokenizeResult {
    const cls = self.string_to_idx.get("[CLS]") orelse return error.NoCls;
    const sep = self.string_to_idx.get("[SEP]") orelse return error.NoSep;
    const unk = self.string_to_idx.get("[UNK]") orelse return error.NoUnk;

    const max_token_len = 512;
    const input_base = @intFromPtr(input_text.ptr);

    var ret = std.ArrayList(i64).empty;
    var offsets = std.ArrayList(TokenOffset).empty;
    try ret.append(alloc, cls);
    try offsets.append(alloc, .{ .start = 0, .end = 0 });

    var candidate_it = CandidateIter{
        .buf = input_text,
        .idx = 0,
    };

    while (candidate_it.next()) |candidate_in| {
        if (ret.items.len > max_token_len) break;
        if (candidate_in.len == 0) continue;

        // All candidates are subslices of input_text, so ptr arithmetic gives
        // the byte range of this candidate in the source.
        const candidate_start: u32 = @intCast(@intFromPtr(candidate_in.ptr) - input_base);
        const candidate_end: u32 = candidate_start + @as(u32, @intCast(candidate_in.len));

        var candidate_buf: [4096]u8 = undefined;

        // We're being a little clever with this buffer. As we lowercase the
        // text, we put it into the buffer 2 spots in. This means no matter how
        // much of the first part of the text we consume, we can replace the
        // end of it with ##, allowing the remaining text to have the
        // ##<remaining> data as a contiguous chunk
        _ = std.ascii.lowerString(candidate_buf[2..], candidate_in);
        const buf_end = 2 + candidate_in.len;

        // Start and end track what area of the buf we're looking at
        var start: usize = 2;
        var end: usize = buf_end;

        var state: enum {
            full_word,
            continuation,

            fn isFinished(state: @This(), start_idx: usize, end_idx: usize) bool {
                const diff = end_idx - start_idx;
                return switch (state) {
                    .full_word => diff == 0,
                    // When we are handling continuations, we SHOULD NOT try to
                    // match ## or # into the vocab
                    .continuation => diff <= 2,
                };
            }
        } = .full_word;

        while (!state.isFinished(start, end)) {
            // No need to crunch more numbers if we have too many tokens
            if (ret.items.len > max_token_len) break;

            const token = self.string_to_idx.get(candidate_buf[start..end]) orelse {
                end -= 1;
                continue;
            };

            try ret.append(alloc, token);
            try offsets.append(alloc, .{ .start = candidate_start, .end = candidate_end });

            @memset(candidate_buf[end - 2 .. end], '#');
            start = end - 2;
            end = buf_end;
            state = .continuation;
        }

        if (!state.isFinished(start, buf_end)) {
            try ret.append(alloc, unk);
            try offsets.append(alloc, .{ .start = candidate_start, .end = candidate_end });
        }
    }

    try ret.append(alloc, sep);
    try offsets.append(alloc, .{ .start = @intCast(input_text.len), .end = @intCast(input_text.len) });

    if (ret.items.len > max_token_len) {
        ret.shrinkRetainingCapacity(max_token_len);
        ret.items[max_token_len - 1] = sep;
        offsets.shrinkRetainingCapacity(max_token_len);
        offsets.items[max_token_len - 1] = .{ .start = @intCast(input_text.len), .end = @intCast(input_text.len) };
    }

    return .{ .tokens = ret.items, .offsets = offsets.items };
}

const CandidateIter = struct {
    buf: []const u8,
    idx: usize,

    pub fn next(self: *CandidateIter) ?[]const u8 {
        if (self.buf.len == 0) return null;
        while (self.idx < self.buf.len) {
            switch (categorizeChar(self.buf[self.idx], self.idx)) {
                .none => self.idx += 1,
                .return_buf_and_skip => {
                    defer self.consume(self.idx + 1);
                    return self.buf[0..self.idx];
                },
                .return_buf_and_keep => {
                    defer self.consume(self.idx);
                    return self.buf[0..self.idx];
                },
                .return_buf_including_char => {
                    defer self.consume(self.idx + 1);
                    return self.buf[0 .. self.idx + 1];
                },
            }
        }

        defer self.consume(self.idx);
        return self.buf;
    }

    const CharAction = enum {
        none,
        return_buf_and_skip,
        return_buf_and_keep,
        return_buf_including_char,
    };

    fn categorizeChar(c: u8, idx: usize) CharAction {
        if (std.ascii.isWhitespace(c)) {
            // Whitespace is not interesting
            return .return_buf_and_skip;
        } else if (isPunctuation(c)) {
            if (idx == 0) {
                // Punctuation is it's own token
                return .return_buf_including_char;
            } else {
                // But it is also a delimiter
                return .return_buf_and_keep;
            }
        }
        return .none;
    }

    fn isPunctuation(cp: u8) bool {
        // https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L386
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126))
        {
            return true;
        }

        return false;
    }

    fn consume(self: *CandidateIter, amount: usize) void {
        if (amount >= self.buf.len) {
            self.buf = "";
        } else {
            self.buf = self.buf[amount..];
        }
        self.idx = 0;
    }
};

fn loadVocab(arena: std.mem.Allocator, expansion: sphtud.util.ExpansionAlloc) !StringToIdx {
    var r = std.Io.Reader.fixed(minilm_vocab);

    var ret = try sphtud.util.StringHashMap(usize).init(
        arena,
        expansion,
        32 * 1024,
        64 * 1024 * 8, // our vocab was ~30k. 16x seems like good headroom
    );

    var idx: usize = 0;
    while (try r.takeDelimiter('\n')) |line| {
        try ret.put(try arena.dupe(u8, line), idx);
        idx += 1;
    }

    return ret;
}

fn checkExpected(tokenized: []const usize, expected: []const usize) !void {
    if (tokenized.len != expected.len) return error.No;
    for (tokenized, expected) |a, b| {
        if (a != b) return error.No;
    }
}

fn parseExpected(alloc: std.mem.Allocator, r: *std.Io.Reader) ![]const i64 {
    var ret = std.ArrayList(i64).empty;

    while (try r.takeDelimiter(' ')) |text| {
        try ret.append(alloc, try std.fmt.parseInt(i64, text, 10));
    }
    return ret.items;
}

const samples: []const [2][]const u8 = &.{
    .{ @embedFile("res/tokenizer/Iron_Man_2_10"), @embedFile("res/tokenizer/Iron_Man_2_10.tokenized") },
    .{ @embedFile("res/tokenizer/It_Takes_Two_(1995_film)_11"), @embedFile("res/tokenizer/It_Takes_Two_(1995_film)_11.tokenized") },
    .{ @embedFile("res/tokenizer/The_Lego_Batman_Movie_10"), @embedFile("res/tokenizer/The_Lego_Batman_Movie_10.tokenized") },
    .{ @embedFile("res/tokenizer/2BR02B:_To_Be_or_Naught_to_Be_14"), @embedFile("res/tokenizer/2BR02B:_To_Be_or_Naught_to_Be_14.tokenized") },
};

test "tokenization matches python" {
    var alloc = sphtud.alloc.BufAllocator.init(try std.heap.page_allocator.alloc(u8, 10 * 1024 * 1024));
    defer std.heap.page_allocator.free(alloc.buf);

    const scratch = alloc.backLinear();

    const tokenizer = try Tokenizer.init(alloc.allocator());

    for (samples) |item| {
        const input_text = item[0];
        const expected_text = item[1];

        var expected_r = std.Io.Reader.fixed(expected_text);
        const expected = try parseExpected(scratch.allocator(), &expected_r);

        const tokenized = try tokenizer.tokenize(scratch.allocator(), input_text);

        try std.testing.expectEqualSlices(i64, expected, tokenized.tokens);
    }
}
