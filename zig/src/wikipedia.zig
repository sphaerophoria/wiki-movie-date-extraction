const std = @import("std");
const sphtud = @import("sphtud");
pub const TemplateIterator = struct {
    buf: sphtud.lex.Buf,
    state: union(enum) {
        parsing,
        finished,
    },

    pub fn init(data: []const u8) !TemplateIterator {
        const buf = sphtud.lex.Buf.init(data);
        return .initLex(buf);
    }

    pub fn initLex(buf_: sphtud.lex.Buf) !TemplateIterator {
        var buf = buf_;
        _ = buf.takeSequence("{{") orelse return error.NotATemplate;
        return .{
            .buf = buf,
            .state = .parsing,
        };
    }

    pub fn next(self: *TemplateIterator) !?[]const u8 {
        // if nested {{ count == 0, look for |
        // otherwise look for }}

        switch (self.state) {
            .parsing => {},
            .finished => return null,
        }

        var tmp = self.buf.tmp();

        const Tag = enum {
            @"{{",
            @"[[",
        };

        const max_depth = 20;
        var tag_stack_buf: [max_depth]Tag = undefined;
        var tag_stack = std.ArrayList(Tag).initBuffer(&tag_stack_buf);

        while (true) {
            _ = tmp.takeUntilAny("{[|}]");
            const idx = tmp.takeOne("{[|}]") orelse return error.EndOfStream;
            switch (idx.data(tmp)) {
                '{' => {
                    _ = tmp.takeOne("{") orelse continue;
                    //std.debug.print("Found {{{{ at {s}\n", .{tmp.remaining()[0..20]});
                    try tag_stack.appendBounded(.@"{{");
                },
                '[' => {
                    _ = tmp.takeOne("[") orelse continue;
                    //std.debug.print("Found [[ at {s}\n", .{tmp.remaining()[0..20]});
                    try tag_stack.appendBounded(.@"[[");
                },
                '}' => {
                    _ = tmp.takeOne("}") orelse continue;

                    if (tag_stack.items.len == 0) {
                        var range = self.buf.commit(tmp).?;
                        range.end -= 2;
                        const ret = range.data(self.buf);
                        self.state = .finished;
                        return ret;
                    } else {
                        if (tag_stack.getLastOrNull() != .@"{{") continue;
                        _ = tag_stack.pop();
                    }
                },
                ']' => {
                    if (tag_stack.getLastOrNull() != .@"[[") continue;
                    _ = tmp.takeOne("]") orelse continue;
                    _ = tag_stack.pop();
                },
                '|' => {
                    if (tag_stack.items.len != 0) continue;

                    var ret = self.buf.commit(tmp).?;
                    ret.end -= 1;
                    return ret.data(self.buf);
                },
                else => unreachable,
            }
        }
    }
};

test "TemplateIterator film date" {
    var it = try TemplateIterator.init("{{Film date|2026|12|10}}");

    try std.testing.expectEqualStrings("Film date", try it.next() orelse unreachable);
    try std.testing.expectEqualStrings("2026", try it.next() orelse unreachable.?);
    try std.testing.expectEqualStrings("12", try it.next() orelse unreachable.?);
    try std.testing.expectEqualStrings("10", try it.next() orelse unreachable.?);
    try std.testing.expectEqual(null, try it.next());
}

test "TemplateIterator nested list" {
    var it = try TemplateIterator.init("{{Something|{{a | b}}}}");

    try std.testing.expectEqualStrings("Something", try it.next() orelse unreachable);
    try std.testing.expectEqualStrings("{{a | b}}", try it.next() orelse unreachable.?);
    try std.testing.expectEqual(null, try it.next());
}

test "TemplateIterator square brackets list" {
    var it = try TemplateIterator.init("{{Something|[[a | b]]}}");

    try std.testing.expectEqualStrings("Something", try it.next() orelse unreachable);
    try std.testing.expectEqualStrings("[[a | b]]", try it.next() orelse unreachable.?);
    try std.testing.expectEqual(null, try it.next());
}
