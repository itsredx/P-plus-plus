; ModuleID = "main"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @"main"()
{
entry:
  %"strptr" = bitcast [11 x i8]* @"str_const" to i8*
  %"calltmp" = call i32 @"puts"(i8* %"strptr")
  store double 0x402e000000000000, double* @"x"
  %"fmtptr" = bitcast [4 x i8]* @"print_fmt" to i8*
  %"calltmp.1" = call i32 (i8*, ...) @"printf"(i8* %"fmtptr", double 0x4028000000000000)
  %"loadtmp" = load double, double* @"x"
  %"faddtmp" = fadd double 0x4008000000000000, %"loadtmp"
  %"fmtptr.1" = bitcast [4 x i8]* @"print_fmt" to i8*
  %"calltmp.2" = call i32 (i8*, ...) @"printf"(i8* %"fmtptr.1", double %"faddtmp")
  store double 0x4008000000000000, double* @"y"
  %"loadtmp.1" = load double, double* @"y"
  %"fmtptr.2" = bitcast [4 x i8]* @"print_fmt" to i8*
  %"calltmp.3" = call i32 (i8*, ...) @"printf"(i8* %"fmtptr.2", double %"loadtmp.1")
  ret i32 0
}

@"str_const" = internal constant [11 x i8] c"helo world\00"
declare i32 @"puts"(i8* %".1")

@"x" = internal global double              0x0
@"print_fmt" = internal constant [4 x i8] c"%f\0a\00"
declare i32 @"printf"(i8* %".1", ...)

@"y" = internal global double              0x0