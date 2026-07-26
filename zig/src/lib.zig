pub const Tokenizer = @import("Tokenizer.zig");
pub const Preprocessor = @import("Preprocessor.zig");
pub const Date = @import("Date.zig");
pub const Onnx = @import("Onnx.zig");

const std = @import("std");

// FIXME: Probably lives in it's own file
pub const DateResolver = struct {
    onnx: Onnx,
    session: Onnx.Session,
    tokenizer: Tokenizer,

    const model = @embedFile("model.int8.onnx");

    pub fn init(arena: std.mem.Allocator) !DateResolver {
        const tokenizer = try Tokenizer.init(arena);

        const onnx = try Onnx.init();
        errdefer onnx.deinit();

        const session = try onnx.loadModel(model);
        errdefer session.deinit(onnx);

        return .{
            .onnx = onnx,
            .session = session,
            .tokenizer = tokenizer,
        };
    }

    pub fn deinit(self: DateResolver) void {
        self.session.deinit(self.onnx);
        self.onnx.deinit();
    }

    pub fn resolveDateCe(self: DateResolver, scratch: std.mem.Allocator, title: []const u8, page_text: []const u8, theatrical_release_date_ce: ?i64) !?i64 {
        var pp = try Preprocessor.init(scratch, title, page_text);

        const theatrical_release_date_unwrapped = theatrical_release_date_ce orelse 0;
        var mask_buf: [512]i64 = undefined;
        @memset(&mask_buf, 1);

        // Sometimes we make mistakes, gate with theatrical release just in
        // case. I know sometimes movies come out on DVD first and then are
        // released in theaters later, but like, whatever
        var earliest_after_theatrical: i64 = std.math.maxInt(i64);

        while (try pp.next()) |preprocessed| {
            const result = try self.tokenizer.tokenize(scratch, preprocessed);

            var onnx_buf: [4096]u8 = undefined;
            var onnx_alloc = try Onnx.Allocator.init(&onnx_buf);
            defer onnx_alloc.deinit(self.onnx);

            var dims = [2]i64{ 1, @intCast(result.tokens.len) };
            const ids = try self.onnx.asTensor(i64, &onnx_alloc, result.tokens, dims[0..]);
            const mask = try self.onnx.asTensor(i64, &onnx_alloc, mask_buf[0..result.tokens.len], dims[0..]);

            const outputs = try self.session.run(
                self.onnx,
                &onnx_alloc,
                &.{ "input_ids", "attention_mask" },
                &.{ ids, mask },
                &.{ "bios", "mediums" },
            );

            const bios, const mediums = outputs[0..2].*;
            if (!bios.dimsMatch(&.{ 1, null, 3 })) return error.Unexpected;
            if (!mediums.dimsMatch(&.{ 1, null, 2 })) return error.Unexpected;

            var span_it = SpanIter{
                .source = preprocessed,
                .offsets = result.offsets,
                .bios = try bios.asSlice(f32, self.onnx),
                .mediums = try mediums.asSlice(f32, self.onnx),
                .pos = 0,
            };

            while (span_it.next()) |span| {
                if (!span.home) continue;
                const parsed_date = Date.parse(span.text);
                const parsed_ce_day = parsed_date.toCeDay() catch continue;
                std.debug.print("{s} ({any} {d}) ({d})\n", .{ span.text, parsed_date, parsed_date.toCeDay() catch 0, span.home_confidence });
                if (parsed_ce_day < theatrical_release_date_unwrapped) continue;
                earliest_after_theatrical = @min(earliest_after_theatrical, parsed_ce_day);
            }
        }

        if (earliest_after_theatrical == std.math.maxInt(i64)) return null;
        return earliest_after_theatrical;
    }
};

// FIXME: Probably lives in it's own file
const SpanIter = struct {
    source: []const u8,
    offsets: []const Tokenizer.TokenOffset,
    bios: []const f32,
    mediums: []const f32,
    pos: usize,

    const Span = struct {
        text: []const u8,
        home: bool,
        theater: bool,
        home_confidence: f32,
    };

    pub fn next(self: *SpanIter) ?Span {
        const num_tokens = self.bios.len / 3;

        if (!self.takeUntilBeginning(num_tokens)) return null;

        // self.pos is at a .beginning tag
        const medium_logits = self.currentMediums();
        const text_start = self.offsets[self.pos].start;

        // Consume forward into first I tag
        self.pos += 1;

        self.takeWhileInside(num_tokens);

        const text_end = self.offsets[self.pos - 1].end;

        return .{
            .text = self.source[text_start..text_end],
            .home = medium_logits[0] > 0,
            .theater = medium_logits[1] > 0,
            .home_confidence = medium_logits[0],
        };
    }

    fn takeUntilBeginning(self: *SpanIter, num_tokens: usize) bool {
        while (true) {
            if (self.pos >= num_tokens) return false;
            if (self.bio() == .beginning) return true;
            self.pos += 1;
        }
    }

    fn takeWhileInside(self: *SpanIter, num_tokens: usize) void {
        while (true) {
            if (self.pos >= num_tokens) return;
            if (self.bio() != .inside) return;
            self.pos += 1;
        }
    }

    fn bio(self: *SpanIter) Bio {
        return .fromNnOutput(self.bios[self.pos * 3..][0..3]);
    }

    fn currentMediums(self: *SpanIter) *const [2]f32 {
        return self.mediums[self.pos * 2..][0..2];
    }

    const Bio = enum {
        outside,
        beginning,
        inside,

        fn fromNnOutput(vals: *const [3]f32) Bio {
            switch (std.mem.indexOfMax(f32, vals)) {
                0 => return .outside,
                1 => return .beginning,
                2 => return .inside,
                else => unreachable,
            }
        }
    };
};

test {
    std.testing.refAllDecls(@This());
}
