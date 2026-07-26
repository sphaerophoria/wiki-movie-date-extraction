const std = @import("std");

pub fn build(b: *std.Build) !void {
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{});


    const asset_dir = b.option([]const u8, "assets", "local asset override");

    const sphtud = b.createModule(.{
        .root_source_file = b.path("sphtud/src/sphtud.zig"),
    });

    const get_assets = b.addSystemCommand(&.{
        "sphasset/sphasset",
        "unpack",
        "v2",
    });

    const assets = if (asset_dir) |d| b.path(d) else get_assets.addOutputDirectoryArg("assets");
    const model_p = assets.path(b, "model.int8.onnx");
    const model = b.createModule(.{
        .root_source_file = model_p,
    });

    const exe = b.addExecutable(.{
        .name = "preprocess",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/preprocess.zig"),
            .target = target,
            .optimize = .Debug,
        }),
    });
    exe.root_module.addImport("sphtud", sphtud);
    b.installArtifact(exe);

    const film_index = b.addExecutable(.{
        .name = "film_index",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/film_index.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    film_index.root_module.addImport("sphtud", sphtud);
    b.installArtifact(film_index);

    const onnx = b.addTranslateC(.{
        .root_source_file = b.path("src/onnx.h"),
        .target = target,
        .optimize = optimize,
    });
    onnx.linkSystemLibrary("onnxruntime", .{});

    const run_model = b.addExecutable(.{
        .name = "run_model",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/run_model.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    run_model.root_module.addImport("sphtud", sphtud);
    run_model.root_module.addImport("onnx", onnx.createModule());
    run_model.root_module.link_libc = true;
    run_model.root_module.addImport("model.int8.onnx", model);
    run_model.use_llvm = false;
    b.installArtifact(run_model);

    const tests = b.addTest(.{
        .name = "tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = .Debug,
        }),
    });
    tests.root_module.addImport("sphtud", sphtud);
    b.installArtifact(tests);

    const lib = b.addModule("home_release_resolver", .{
        .root_source_file = b.path("src/lib.zig"),
        .optimize = optimize,
        .target = target,
    });
    lib.addImport("sphtud", sphtud);
    lib.addImport("onnx", onnx.createModule());
    lib.addImport("model.int8.onnx", model);
    lib.link_libc = true;
}
