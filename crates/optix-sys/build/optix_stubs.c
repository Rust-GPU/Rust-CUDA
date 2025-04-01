// Optix uses a function table approach to load its api at runtime. The table is defined
// by including optix_function_table_definition.h here.
#include "optix_function_table_definition.h"

// These are copied from optix_stubs.h so they do not get affected by the redefinition
// of inline. See below.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
// For convenience the library is also linked in automatically using the #pragma command.
#include <cfgmgr32.h>
#pragma comment(lib, "Cfgmgr32.lib")
#include <string.h>
#else
#include <dlfcn.h>
#endif

// optix_stubs.h contains the functions needed to load the library and provides stubs
// to call the functions in the function table. However, the stubs are defined as
// `inline`, and won't be included in the final binary as is. We work around this by
// redefining `inline` to do nothing before including the header.
#define inline
#include "optix_stubs.h"
#undef inline
