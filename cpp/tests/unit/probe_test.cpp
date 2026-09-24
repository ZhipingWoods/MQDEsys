#include <cstring>

#include <gtest/gtest.h>

#include "mqde_probe.h"

TEST(probe, always_finds_cpu) {
  mqde_device_info devs[16];
  int n = 0;
  ASSERT_EQ(mqde_probe_all(devs, 16, &n), 0);
  ASSERT_GE(n, 1);
  EXPECT_EQ(devs[0].kind, MQDE_DEV_CPU);
  EXPECT_GT(std::strlen(devs[0].name), 0u);
  EXPECT_GT(devs[0].compute_units, 0u);
  for (int i = 1; i < n; ++i) {
    EXPECT_EQ(devs[i].kind, MQDE_DEV_GPU);
    EXPECT_GT(devs[i].mem_total_bytes, 0u);
  }
}

TEST(probe, rejects_bad_arguments) {
  int n = 0;
  EXPECT_NE(mqde_probe_all(nullptr, 4, &n), 0);
  EXPECT_GT(std::strlen(mqde_probe_last_error()), 0u);
}
