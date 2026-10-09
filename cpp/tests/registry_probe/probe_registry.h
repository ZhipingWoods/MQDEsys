#pragma once
#include "mqde/common/registry.h"
struct Shape {
  virtual ~Shape() = default;
  virtual int sides() const = 0;
};
mqde::Registry<Shape>& shapes();
