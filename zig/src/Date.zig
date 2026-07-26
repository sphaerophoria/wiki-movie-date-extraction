const std = @import("std");
const sphtud = @import("sphtud");

day: ?u8 = null,
month: ?u8 = null,
year: ?u16 = null,

const Date = @This();

pub fn parse(input: []const u8) Date {
    var lower_buf: [256]u8 = undefined;
    const lower = std.ascii.lowerString(&lower_buf, input);

    const month = extractMonth(lower);

    var buf = sphtud.lex.Buf.init(input);

    takeUntilDigit(&buf);
    const num_a = number(&buf);

    takeUntilDigit(&buf);
    const num_b = number(&buf);

    var day: ?u8 = null;
    var year: ?u16 = null;

    switch (DayOrYear.parse(num_a, buf)) {
        .day => |d| day = d,
        .year => |y| year = y,
        .neither => {},
    }

    switch (DayOrYear.parse(num_b, buf)) {
        .day => |d| day = d,
        .year => |y| year = y,
        .neither => {},
    }

    if (month == null) {
        // Day by itself, or (day, year) both don't make sense
        day = null;
    }

    return .{
        .day = day,
        .month = month,
        .year = year,
    };
}

pub fn toCeDay(self: Date) !i64 {
    const year = self.year orelse return error.NoYear;
    const month: sphtud.datetime.Month = @enumFromInt((self.month orelse 12) - 1);
    const sphtud_date = sphtud.datetime.Date {
        .day = self.day orelse sphtud.datetime.daysInMonth(year, month), // fixme what if it's a 28 day or 30 day month
        .month = month,
        .year = self.year orelse return error.NoYear,
    };
    return sphtud_date.toCeDay();
}

pub const month_names = [12][]const u8{
    "january", "february", "march", "april", "may", "june",
    "july",    "august",   "september", "october", "november", "december",
};

pub const month_abbrevs = [12][]const u8{
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
};

// Max days per month; February allows 29 (ignoring leap year precision).
const max_days_per_month = [12]u8{ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

fn takeUntilDigit(buf: *sphtud.lex.Buf) void {
    for (buf.idx..buf.data.len) |i| {
        if (buf.data[i] >= '0' and buf.data[i] <= '9') {
            buf.idx = i;
            return;
        }
    }

    buf.idx = buf.data.len;
}

fn number(buf: *sphtud.lex.Buf) ?sphtud.lex.Range {
    var tmp = buf.tmp();
    while (tmp.takeOneBetween('0', '9') != null) {}
    return buf.commit(tmp);
}

fn extractMonth(input: []const u8) ?u8 {
    for (month_names, 1..) |name, idx| {
        if (std.mem.indexOf(u8, input, name) != null) return @intCast(idx);
    }

    for (month_abbrevs, 1..) |name, idx| {
        if (std.mem.indexOf(u8, input, name) != null) return @intCast(idx);
    }

    return null;
}


const DayOrYear = union(enum) {
    day: u8,
    year: u16,
    neither,

    fn parse(range_opt: ?sphtud.lex.Range, buf: sphtud.lex.Buf) DayOrYear {
        const range = range_opt orelse return .neither;
        const parsed = std.fmt.parseInt(u16, range.data(buf), 10) catch return .neither;
        if (parsed > 1800 and parsed < 3000) return .{ .year = parsed };
        if (parsed > 0 and parsed < 32) return .{ .day = @intCast(parsed) };
        return .neither;
    }
};

test "full date with ordinal" {
    const d = parse("20th of January, 2026");
    try std.testing.expectEqual(@as(?u8, 20), d.day);
    try std.testing.expectEqual(@as(?u8, 1), d.month);
    try std.testing.expectEqual(@as(?u16, 2026), d.year);
}

test "month only" {
    const d = parse("March");
    try std.testing.expectEqual(@as(?u8, null), d.day);
    try std.testing.expectEqual(@as(?u8, 3), d.month);
    try std.testing.expectEqual(@as(?u16, null), d.year);
}

test "month and year no day" {
    const d = parse("September 1995");
    try std.testing.expectEqual(@as(?u8, null), d.day);
    try std.testing.expectEqual(@as(?u8, 9), d.month);
    try std.testing.expectEqual(@as(?u16, 1995), d.year);
}

test "abbreviation with day and year" {
    const d = parse("Jan 15, 2020");
    try std.testing.expectEqual(@as(?u8, 15), d.day);
    try std.testing.expectEqual(@as(?u8, 1), d.month);
    try std.testing.expectEqual(@as(?u16, 2020), d.year);
}

test "year only" {
    const d = parse("2026");
    try std.testing.expectEqual(@as(?u8, null), d.day);
    try std.testing.expectEqual(@as(?u8, null), d.month);
    try std.testing.expectEqual(@as(?u16, 2026), d.year);
}

test "no date" {
    const d = parse("no date here");
    try std.testing.expectEqual(@as(?u8, null), d.day);
    try std.testing.expectEqual(@as(?u8, null), d.month);
    try std.testing.expectEqual(@as(?u16, null), d.year);
}

test "day before month in text" {
    const d = parse("3rd of December");
    try std.testing.expectEqual(@as(?u8, 3), d.day);
    try std.testing.expectEqual(@as(?u8, 12), d.month);
    try std.testing.expectEqual(@as(?u16, null), d.year);
}

test "sept abbreviation" {
    const d = parse("sept 2001");
    try std.testing.expectEqual(@as(?u8, 9), d.month);
    try std.testing.expectEqual(@as(?u16, 2001), d.year);
}
