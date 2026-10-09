/* test plugin: dup_a */
#include "mqde/hal/mqde_hal.h"
#include <string.h>
#include <stdio.h>
__attribute__((unused)) static const char* le(void){return "boom";}
__attribute__((unused)) static int probe1(mqde_hal_device*o,int m,int*n){(void)m;memset(o,0,sizeof*o);o->kind=1;snprintf(o->vendor,32,"acme");snprintf(o->name,128,"Acme One");snprintf(o->arch,32,"a1");o->mem_total_bytes=1<<30;*n=1;return 0;}
static const mqde_hal_plugin_v1 P={.abi_version=MQDE_HAL_ABI_VERSION,.struct_size=sizeof(mqde_hal_plugin_v1),.name="dup",.vendor="v",.plugin_version="1",.device_probe=probe1,.last_error=le};
MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void){return &P;}
