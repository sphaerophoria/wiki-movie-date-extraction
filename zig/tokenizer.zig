const std = @import("std");
const sphtud = @import("sphtud");

const Vocab = sphtud.util.StringHashMap(usize);

const CandidateIter = struct {
    buf: []const u8,
    idx: usize,

    pub fn next(self: *CandidateIter) ?[]const u8 {
        if (self.buf.len == 0) return null;
        while (self.idx < self.buf.len) {
            const c = self.buf[self.idx];
              //FIXME: unicode lol
            if (std.ascii.isWhitespace(c)) {
                // Return up to ws, but consume ws for later
                defer self.consume(self.idx + 1);
                return self.buf[0..self.idx];
            } else if (isPunctuation(c)) {
                if (self.idx == 0) {
                    // We are the punctuation, export as single char
                    defer self.consume(1);
                    return self.buf[0..1];
                } else {
                    // Consume up until the punctuation, leaving it behind for next iter
                    defer self.consume(self.idx);
                    return self.buf[0..self.idx];
                }
            }
            self.idx += 1;
        }
        defer self.consume(self.idx);
        return self.buf;
    }

    fn isPunctuation(cp: u8) bool {
      if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
          (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)) {
          return true;
      }

      //FIXME: unicode lol
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

fn loadVocab(arena: std.mem.Allocator, expansion: sphtud.util.ExpansionAlloc, vocab_path: [:0]const u8) !Vocab {
    const f = try sphtud.io.open(vocab_path, .{}, 0);
    defer sphtud.io.close(f);

    var read_buf: [4096]u8 = undefined;
    var r = sphtud.io.Reader.init(f, &read_buf);

    var ret = try sphtud.util.StringHashMap(usize).init(
        arena,
        expansion,
        32 * 1024,
        64 * 1024 * 8, // our vocab was ~30k. 16x seems like good headroom
    );

    var idx: usize = 0;
    while (try r.interface.takeDelimiter('\n')) |line| {
        try ret.put(try arena.dupe(u8, line), idx);
        idx += 1;
    }

    return ret;
}

pub fn tokenize(alloc: std.mem.Allocator, input_text: []const u8, vocab: Vocab) ![]usize {
    const cls = vocab.get("[CLS]") orelse return error.NoCls;
    const sep = vocab.get("[SEP]") orelse return error.NoSep;
    const unk = vocab.get("[UNK]") orelse return error.NoUnk;

    var ret = std.ArrayList(usize).empty;
    try ret.append(alloc, cls);
    var candidate_it = CandidateIter {
        .buf = input_text,
        .idx = 0,
    };
    while (candidate_it.next()) |candidate_in| {
        if (candidate_in.len == 0) continue;
        var candidate_buf: [4096]u8 = undefined;
        candidate_buf[0] = '#';
        candidate_buf[1] = '#';

        var candidate = std.ascii.lowerString(candidate_buf[2..], candidate_in);
        var min_end: usize = 0;
        std.debug.print("Candidate {s}\n", .{candidate});

        var success = false;
        var end: usize = candidate.len;
        while (candidate.len > 0) {
            std.debug.print("{s}\n", .{candidate});
            const token = vocab.get(candidate[0..end]) orelse {
                if (end == min_end) break;
                end -= 1;
                continue;
            };

            std.debug.print("Found {s} -> {d}\n", .{candidate[0..end], token});
            try ret.append(alloc, token);
            if (end == candidate.len) {
                success = true;
                break;
            }


            const new_segment = candidate[end..];
            if (new_segment.len == 0) {
                break;
            }
            @memmove(candidate_buf[2..new_segment.len + 2], new_segment);
            candidate = candidate_buf[0..new_segment.len + 2];
            end = candidate.len;
            min_end = 2;
        }

        if (!success) {
            try ret.append(alloc, unk);
        }
    }
    try ret.append(alloc, sep);

    return ret.items;
}
fn checkExpected(tokenized: []const usize, expected: []const usize) !void {
    if (tokenized.len != expected.len) return error.No;
    for (tokenized, expected) |a, b| {
        if (a != b) return error.No;
    }
}

fn parseExpected(alloc: std.mem.Allocator, file: c_int) ![]const usize {
    var ret = std.ArrayList(usize).empty;

    var buf: [4096]u8 = undefined;
    var r = sphtud.io.Reader.init(file, &buf);
    while (try r.interface.takeDelimiter(' ')) |text| {
        try ret.append(alloc, try std.fmt.parseInt(usize, text, 10));
    }
    return ret.items;
}

pub fn main(init: std.process.Init.Minimal) !void {
    std.debug.print("{d}\n", .{'–'});
    if (true) return;
    var alloc = sphtud.alloc.BufAllocator.init(try std.heap.page_allocator.alloc(u8, 10 * 1024 * 1024));
    const scratch = alloc.backLinear();

    var args = init.args.iterate();
    _ = args.next();

    const vocab_path = args.next() orelse return error.NoVocab;
    const input_dir_path = args.next() orelse return error.NoText;

    const dir = try sphtud.io.open(input_dir_path, .{ .DIRECTORY = true }, 0);
    defer sphtud.io.close(dir);

    var diriter_buf: [4096]u8 align(8) = undefined;
    var dir_it = sphtud.io.DirIter.init(dir, &diriter_buf);

    const vocab = try loadVocab(alloc.allocator(), .linear(alloc.allocator()), vocab_path);

    while (try dir_it.next()) |entry| {
        const cp = scratch.checkpoint();
        defer scratch.restore(cp);

        if (std.mem.endsWith(u8, entry.name, ".tokenized")) {
            continue;
        }

        if (std.mem.eql(u8, entry.name, ".")) {
            continue;
        }
        if (std.mem.eql(u8, entry.name, "..")) {
            continue;
        }

        std.debug.print("opening {s}\n", .{entry.name});
        const content_file = try sphtud.io.openat(dir, entry.name, .{}, 0);
        defer sphtud.io.close(content_file);
        const content = blk: {
            var r = sphtud.io.Reader.init(content_file, &.{});
            break :blk try r.interface.allocRemaining(scratch.allocator(), .unlimited);
        };
        std.debug.print("Tokenizing\n", .{});
        const tokenized = try tokenize(scratch.allocator(), content, vocab);
        std.debug.print("Tokenizing done\n", .{});

        const expected_tokens_path = try std.fmt.allocPrintSentinel(scratch.allocator(), "{s}.tokenized", .{entry.name}, 0);
        const expected_tokens_file = try sphtud.io.openat(dir, expected_tokens_path, .{}, 0);
        defer sphtud.io.close(expected_tokens_file);

        std.debug.print("Expected\n", .{});
        const expected = try parseExpected(scratch.allocator(), expected_tokens_file);

        checkExpected(tokenized, expected) catch |e| {
            std.debug.print("UH OH {s}\n{any}\n{any}\n", .{entry.name, tokenized, expected});
            return e;
        };

    }

    //const unk = vocab.get("[UNK]") orelse return error.NoUnk;


    // Start with [CLS]
    // FIXME: assuming ascii
    // TODO: What subset of characters are valid in wikitext
    //var candidate_it = std.mem.splitAny(u8, input_text, &std.ascii.whitespace);
    //while (candidate_it.next()) |candidate_in| {
    //    var candidate_buf: [4096]u8 = undefined;
    //    var candidate = candidate_in;

    //    var success = false;
    //    var end: usize = candidate.len;
    //    while (candidate.len > 0) {

    //        const token = vocab.get(candidate[0..end]) orelse {
    //            end -= 1;
    //            continue;
    //        };

    //        std.debug.print("{s} -> ({d})\n", .{candidate[0..end], token});
    //        if (end == candidate.len) {
    //            success = true;
    //            break;
    //        }
    //        candidate = try std.fmt.bufPrint(&candidate_buf, "##{s}", .{candidate[end..]});
    //        end = candidate.len;
    //    }

    //    if (!success) {
    //        std.debug.print("[UNK] -> ({d})\n", .{unk});
    //    }
    //    // Take full range  of candidate
    //    // Check against vocab
    //    // Decrease end idx by 1 until we are in the vocab
    //    // start sub-sequence with ## and go again
    //}
}
