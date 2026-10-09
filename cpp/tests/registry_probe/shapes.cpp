// Lives in a STATIC library and is referenced by nothing except MQDE_FORCE_LINK(square) in the test: the case
// the linker drops.
#include "probe_registry.h"
mqde::Registry<Shape>& shapes() {
  static mqde::Registry<Shape> r;
  return r;
}
struct Square : Shape {
  int sides() const override { return 4; }
};
struct Triangle : Shape {
  int sides() const override { return 3; }
};
MQDE_REGISTER_NAMED(square, shapes(), "square", Square);
MQDE_REGISTER_NAMED(triangle, shapes(), "triangle", Triangle);
