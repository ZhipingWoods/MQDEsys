/* test plugin: returns_null */
#include <stdio.h>
#include <string.h>
#include "mqde/hal/mqde_hal.h"
MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void) { return 0; }
