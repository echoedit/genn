// float->half variants.
// by Fabian "ryg" Giesen.
//
// I hereby place this code in the public domain, as per the terms of the
// CC0 license:
//
//   https://creativecommons.org/publicdomain/zero/1.0/
//
// float_to_half_full: This is basically the ISPC stdlib code, except
// I preserve the sign of NaNs (any good reason not to?)
//
// float_to_half_fast: Almost the same, with some unnecessary cases cut.
//
// float_to_half_fast2: This is where it gets a bit weird. See lengthy
// commentary inside the function code. I'm a bit on the fence about two
// things:
// 1. This *will* behave differently based on whether flush-to-zero is
//    enabled or not. Is this acceptable for ISPC?
// 2. I'm a bit on the fence about NaNs. For half->float, I opted to extend
//    the mantissa (preserving both qNaN and sNaN contents) instead of always
//    returning a qNaN like the original ISPC stdlib code did. For float->half
//    the "natural" thing would be just taking the top mantissa bits, except
//    that doesn't work; if they're all zero, we might turn a sNaN into an
//    Infinity (seriously bad!). I could test for this case and do a sticky
//    bit-like mechanism, but that's pretty ugly. Instead I go with ISPC
//    std lib behavior in this case and just return a qNaN - not quite symmetric
//    but at least it's always safe. Any opinions?
//
// I'll just go ahead and give "fast2" the same treatment as my half->float code,
// but if there's concerns with the way it works I might revise it later, so watch
// this spot.
//
// float_to_half_fast3: Bitfields removed. Ready for SSE2-ification :)
//
// approx_float_to_half: Simpler (but less accurate) version that matches the Fox
// toolkit float->half conversions: http://blog.fox-toolkit.org/?p=40 - note that this
// also (incorrectly) translates some sNaNs into infinity, so be careful!
//
// ----
//
// UPDATE 2016-01-25: Now also with a variant that implements proper round-to-nearest-even.
// It's a bit more expensive and has seen less tweaking than the other variants. On the
// plus side, it doesn't produce subnormal FP32 values as part of generating subnormal
// FP16 values, so the performance is a lot more consistent.
//
// float_to_half_rtne_full: Unoptimized round-to-nearest-break-ties-to-even reference
// implementation.
// 
// float_to_half_fast3_rtne: Variant of float_to_half_fast3 that performs round-to-
// nearest-even.
//
// float_to_half_rtne_SSE2: SSE2 implementation of float_to_half_fast3_rtne.
//
// All three functions have been exhaustively tested to produce the same results on
// all 32-bit floating-point numbers with SSE arithmetic in round-to-nearest-even mode.
// No guarantees for what happens with other rounding modes! (See testbed code.)
//
// ----

#include <cstdint>

//----------------------------------------------------------------------------
// HalfFloat
//----------------------------------------------------------------------------
namespace HalfFloat
{
#ifdef __aarch64__
inline uint16_t __fcvt_f32_f16(float x)
{
    register uint16_t r;

    asm volatile("fcvt %h[r], %s[x]"
                : [r] "=w" (r) : [x] "w" (x));

    return r;
}

inline float __fcvt_f16_f32(uint16_t x)
{
  register float r;

  asm volatile("fcvt %s[r], %h[x]"
              : [r] "=w" (r) : [x] "w" (x));

  return r;
}

#endif  // __aarch64__

union FP32
{
    uint32_t u;
    float f;
    struct
    {
        unsigned int Mantissa : 23;
        unsigned int Exponent : 8;
        unsigned int Sign : 1;
    };
};

union FP16
{
    uint16_t u;
    struct
    {
        unsigned int Mantissa : 10;
        unsigned int Exponent : 5;
        unsigned int Sign : 1;
    };
};

inline uint16_t float_to_half_reference(float in)
{
    FP32 f = {.f = in};
    FP16 o = {.u = 0};

    // Based on ISPC reference code (with minor modifications)
    if (f.Exponent == 255) // Inf or NaN (all exponent bits set)
    {
        o.Exponent = 31;
        o.Mantissa = f.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
    }
    else // Normalized number
    {
        // Exponent unbias the single, then bias the halfp
        const int newexp = f.Exponent - 127 + 15;
        if (newexp >= 31) // Overflow, return signed infinity
            o.Exponent = 31;
        else if (newexp <= 0) // Underflow
        {
            if ((14 - newexp) <= 24) // Mantissa might be non-zero
            {
                const unsigned int mant = f.Mantissa | 0x800000; // Hidden 1 bit
                o.Mantissa = mant >> (14 - newexp);
                if ((mant >> (13 - newexp)) & 1) // Check for rounding
                    o.u++; // Round, might overflow into exp bit, but this is OK
            }
        }
        else
        {
            o.Exponent = newexp;
            o.Mantissa = f.Mantissa >> 13;
            if (f.Mantissa & 0x1000) // Check for rounding
                o.u++; // Round, might overflow to inf, this is OK
        }
    }

    o.Sign = f.Sign;
    return o.u;
}

inline float half_to_float_reference(uint16_t in)
{
    FP32 h = {.u = in};
    static const FP32 magic = {.u = 113 << 23};
    static const unsigned int shifted_exp = 0x7c00 << 13; // exponent mask after shift
    FP32 o;

    o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
    const unsigned int exp = shifted_exp & o.u;   // just the exponent
    o.u += (127 - 15) << 23;        // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
    else if (exp == 0) // Zero/Denormal?
    {
        o.u += 1 << 23;             // extra exp adjust
        o.f -= magic.f;             // renormalize
    }

    o.u |= (h.u & 0x8000) << 16;    // sign bit
    return o.f;
}

inline uint16_t float_to_half(float in)
{
#ifdef __aarch64__
    return __fcvt_f32_f16(in);
#else
    return float_to_half_reference(in);
#endif  
}

inline float half_to_float(uint16_t in)
{
#ifdef __aarch64__
    return __fcvt_f16_f32(in);
#else
    return half_to_float_reference(in);
#endif 
}
}