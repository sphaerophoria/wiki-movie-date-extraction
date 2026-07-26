const c = @import("onnx");
const sphtud = @import("sphtud");
const std = @import("std");

const Onnx = @This();

gort: *const c.OrtApi,
env: *c.OrtEnv,
cpu_memory_info: *c.OrtMemoryInfo,

pub fn init() !Onnx {
    const api_base = c.OrtGetApiBase().*;
    const gort: *const c.OrtApi = api_base.GetApi.?(c.ORT_API_VERSION);

    var env: ?*c.OrtEnv = null;
    try cCheckImpl(gort, gort.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "gort", &env));
    errdefer gort.ReleaseEnv.?(env);

    var cpu_memory_info: ?*c.OrtMemoryInfo = null;
    try cCheckImpl(gort, gort.CreateCpuMemoryInfo.?(c.OrtDeviceAllocator, c.OrtMemTypeCPU, &cpu_memory_info));

    return .{
        .gort = gort,
        .cpu_memory_info = cpu_memory_info orelse return error.Onnx,
        .env = env orelse return error.Onnx,
    };
}

pub fn deinit(self: Onnx) void {
    self.gort.ReleaseMemoryInfo.?(self.cpu_memory_info);
    self.gort.ReleaseEnv.?(self.env);
}

pub fn loadModel(self: Onnx, model_data: []const u8) !Session {
    const gort = self.gort;

    var session_options: ?*c.OrtSessionOptions = null;
    try self.cCheck(gort.CreateSessionOptions.?(&session_options));
    defer gort.ReleaseSessionOptions.?(session_options);

    var session: ?*c.OrtSession = null;
    try self.cCheck(gort.CreateSessionFromArray.?(self.env, model_data.ptr, model_data.len, session_options, &session));

    return .{
        .inner = session.?,
    };
}

pub fn asTensor(self: Onnx, comptime T: type, pool: *Allocator, data: []T, dims: []const i64) !Value {
    var inner: ?*c.OrtValue = null;
    try self.cCheck(self.gort.CreateTensorWithDataAsOrtValue.?(self.cpu_memory_info, data.ptr, data.len * @sizeOf(T), dims.ptr, dims.len, onnxType(T), &inner));

    const ret = Value{ .inner = inner orelse return error.Onnx };
    try pool.append(ret);
    return ret;
}

pub const Allocator = struct {
    alloc: sphtud.alloc.BufAllocator,
    values: sphtud.util.RuntimeSegmentedListUnmanaged(Value),

    // buf is used as the backing buffer for all heap allocations made by this
    // allocator. Luckily those should be relatively few. It's basically used
    // to create slices of inputs/outputs/dims, as well as to track Onnx.Values.
    // A 1k or 4k buffer is not an insane choice here
    pub fn init(buf: []u8) !Allocator {
        var ba = sphtud.alloc.BufAllocator.init(buf);
        const values = try sphtud.util.RuntimeSegmentedListUnmanaged(Value).init(ba.allocator(), ba.expansion(), 8, 16384,);

        return .{
            .alloc = ba,
            .values = values,
        };
    }

    pub fn deinit(self: *Allocator, onnx: Onnx) void {
        var it = self.values.iter();
        while (it.next()) |v| {
            onnx.gort.ReleaseValue.?(v.inner);
        }
    }

    const Checkpoint = struct {
        alloc: usize,
        values: usize,
    };

    pub fn checkpoint(self: *Allocator) Checkpoint {
        return .{
            .alloc = self.alloc.linear().checkpoint(),
            .values = self.values.len,
        };
    }

    pub fn restore(self: *Allocator, onnx: Onnx, cp: Checkpoint) void {
        var it = self.values.iterFrom(cp.values);
        while (it.next()) |value| {
            onnx.gort.ReleaseValue.?(value.inner);
        }

        self.values.shrink(.{
            .alloc = self.alloc.allocator(),
            .info = &.{
                .min_expansion_size_log2 = 0,
                // Typically our allocators do not support free, but in this
                // case we actually can and should because we're about to roll
                // back our allocator
                .supports_free = true,
            },
        }, cp.values);

        self.alloc.linear().restore(cp.alloc);
    }

    pub fn append(self: *Allocator, value: Value) !void {
        try self.values.append(self.alloc.expansion(), value);
    }
};

pub const Session = struct {
    inner: *c.OrtSession,

    pub fn deinit(self: Session, onnx: Onnx) void {
        onnx.gort.ReleaseSession.?(self.inner);
    }

    pub const Output = struct {
        dims: []const i64,
        value: Value,

        pub fn dimsMatch(self: Output, expected: []const ?i64) bool {
            if (expected.len != self.dims.len) return false;

            for (self.dims, expected) |v, e| {
                const v2 = e orelse continue;
                if (v != v2) return false;
            }

            return true;
        }

        pub fn asSlice(self: Output, comptime T: type, onnx: Onnx) ![]T {
            const gort = onnx.gort;
            var output_shape_info: ?*c.OrtTensorTypeAndShapeInfo = null;
            try onnx.cCheck(gort.GetTensorTypeAndShape.?(self.value.inner, &output_shape_info));
            defer onnx.gort.ReleaseTensorTypeAndShapeInfo.?(output_shape_info);

            var elem_type: c.ONNXTensorElementDataType = 0;
            try onnx.cCheck(gort.GetTensorElementType.?(output_shape_info, &elem_type));
            if (elem_type != onnxType(T)) return error.IncorrectType;

            var data: ?[*]T = null;
            try onnx.cCheck(gort.GetTensorMutableData.?(self.value.inner, @ptrCast(&data)));

            var size: usize = @intCast(self.dims[0]);
            for (self.dims[1..]) |dim| {
                size *= @intCast(dim);
            }

            return data.?[0..size];
        }
    };

    pub fn run(
        self: Session,
        onnx: Onnx,
        pool: *Allocator,
        input_names: []const [*:0]const u8,
        inputs: []const Value,
        output_names: []const [*:0]const u8,
    ) ![]Output {
        comptime {
            std.debug.assert(@sizeOf(Value) == @sizeOf(?*c.OrtValue));
            std.debug.assert(@alignOf(Value) == @alignOf(?*c.OrtValue));
        }

        const scratch = pool.alloc.backLinear();

        const cp = scratch.checkpoint();
        defer scratch.restore(cp);

        const outputs = try scratch.allocator().alloc(Value, output_names.len);
        @memset(outputs, .empty);
        std.debug.assert(input_names.len == inputs.len);

        try onnx.cCheck(onnx.gort.Run.?(self.inner, null, input_names.ptr, @ptrCast(inputs.ptr), input_names.len, output_names.ptr, output_names.len, @ptrCast(outputs.ptr)));

        for (outputs) |value| {
            try pool.append(value);
        }

        const ret = try pool.alloc.allocator().alloc(Output, output_names.len);
        for (outputs, ret) |value, *output| {
            var output_shape_info: ?*c.OrtTensorTypeAndShapeInfo = null;
            try onnx.cCheck(onnx.gort.GetTensorTypeAndShape.?(value.inner, &output_shape_info));
            defer onnx.gort.ReleaseTensorTypeAndShapeInfo.?(output_shape_info);

            var out_num_dims: usize = 0;
            try onnx.cCheck(onnx.gort.GetDimensionsCount.?(output_shape_info, &out_num_dims));

            const out_dims = try pool.alloc.allocator().alloc(i64, out_num_dims);
            try onnx.cCheck(onnx.gort.GetDimensions.?(output_shape_info, out_dims.ptr, out_num_dims));
            output.* = .{
                .dims = out_dims,
                .value = value,
            };
        }

        return ret;
    }
};

pub const Value = struct {
    inner: ?*c.OrtValue,

    pub const empty = Value{ .inner = null };
};

fn onnxType(comptime T: type) comptime_int {
    return switch (T) {
        i64 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        f32 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        else => @compileError("Unimplemented onnx type conversion for " ++ @typeName(T)),
    };
}

fn cCheck(self: Onnx, res: ?*c.OrtStatus) !void {
    return cCheckImpl(self.gort, res);
}

fn cCheckImpl(gort: *const c.OrtApi, res: ?*c.OrtStatus) !void {
    const status = res orelse return;
    defer gort.ReleaseStatus.?(status);

    const msg = gort.GetErrorMessage.?(status);
    std.log.err("onnx err: {s}\n", .{msg});

    return error.Onnx;
}
