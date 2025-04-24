#pragma once

#ifdef _WIN32
#if defined(torchish_EXPORTS)
#define TORCHISH_API __declspec(dllexport)
#else
#define TORCHISH_API __declspec(dllimport)
#endif
#else
#define TORCHISH_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define TORCHISH_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define TORCHISH_INLINE_VARIABLE __declspec(selectany)
#else
#define TORCHISH_INLINE_VARIABLE __attribute__((weak))
#endif
#endif