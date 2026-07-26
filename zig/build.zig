const std = @import("std");

pub fn build(b: *std.Build) !void {
    const optimize = b.standardOptimizeOption(.{});
    const sphtud = b.createModule(.{
        .root_source_file = b.path("sphtud/src/sphtud.zig"),
    });

    const exe = b.addExecutable(.{
        .name = "preprocess",
        .root_module = b.createModule(.{
            .root_source_file = b.path("preprocess.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    exe.root_module.addImport("sphtud", sphtud);
    b.installArtifact(exe);

    const tokenizer = b.addExecutable(.{
        .name = "tokenizer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tokenizer.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    tokenizer.root_module.addImport("sphtud", sphtud);
    b.installArtifact(tokenizer);

    const film_index = b.addExecutable(.{
        .name = "film_index",
        .root_module = b.createModule(.{
            .root_source_file = b.path("film_index.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });
    film_index.root_module.addImport("sphtud", sphtud);
    b.installArtifact(film_index);

}
