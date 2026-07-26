const sphtud = @import("sphtud");
const std = @import("std");
const lib = @import("lib.zig");

const Onnx = lib.Onnx;
const Tokenizer = lib.Tokenizer;
const Preprocessor = lib.Preprocessor;


pub fn main(init: std.process.Init.Minimal) !void {
    var arena = sphtud.alloc.BufAllocator.init(try std.heap.page_allocator.alloc(u8, 8 * 1024 * 1024));

    var args = init.args.iterate();
    _ = args.next();

    const input_path = args.next() orelse return error.NoPath;

    const title = if (std.mem.cutLast(u8, input_path, "/")) |v| v[1] else input_path;

    const input = input: {
        const input_f = try sphtud.io.open(input_path, .{}, 0);
        defer sphtud.io.close(input_f);

        var r = sphtud.io.Reader.init(input_f, &.{});
        break :input try r.interface.allocRemaining(arena.allocator(), .unlimited);
    };

    const date_resolver = try lib.DateResolver.init(arena.allocator());
    defer date_resolver.deinit();

    const total_start = try sphtud.io.clock_gettime(.BOOTTIME);

    const date = try date_resolver.resolveDateCe(arena.allocator(), title, input, null);

    const total_end = try sphtud.io.clock_gettime(.BOOTTIME);
    std.debug.print("Took {d}ms\n", .{total_start.durationTo(total_end).toMilliseconds()});

    if (date) |d| {
        const parsed_date = sphtud.datetime.Date.fromCeDay(d);
        std.debug.print("Released on {f}\n", .{parsed_date});
    } else {
        std.debug.print("Unclear\n", .{});
    }
}
