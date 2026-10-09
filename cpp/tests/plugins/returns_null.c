/* test plugin: returns_null */
#include "mqde/hal/mqde_hal.h"
#include <string.h>
#include <stdio.h>
MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void){return 0;}
