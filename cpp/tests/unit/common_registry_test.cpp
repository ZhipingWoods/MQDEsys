// T-0103: the registry, including registrations that live in a static library nothing else references.
#include <gtest/gtest.h>

#include <thread>

#include "mqde/common/registry.h"
#include "probe_registry.h"

MQDE_FORCE_LINK(square);
MQDE_FORCE_LINK(triangle);

namespace {
struct Animal {
  virtual ~Animal() = default;
  virtual std::string noise() const = 0;
};
struct Dog : Animal {
  std::string noise() const override { return "woof"; }
};
}  // namespace

TEST(common_registry, registrations_in_a_static_library_survive_the_linker) {
  EXPECT_TRUE(shapes().contains("square"));
  EXPECT_TRUE(shapes().contains("triangle"));
  auto sq = shapes().create("square");
  ASSERT_TRUE(sq) << sq.error().message;
  EXPECT_EQ((*sq)->sides(), 4);
  EXPECT_EQ(shapes().names(), (std::vector<std::string>{"square", "triangle"}));
}

TEST(common_registry, a_duplicate_name_is_an_error_naming_it_and_keeps_the_first) {
  mqde::Registry<Animal> r;
  ASSERT_TRUE(r.add("dog", [] { return std::make_unique<Dog>(); }));
  auto again = r.add("dog", [] { return std::make_unique<Dog>(); });
  ASSERT_FALSE(again);
  EXPECT_EQ(again.error().code, mqde::Errc::kInvalidArgument);
  EXPECT_NE(again.error().message.find("'dog' is already registered"), std::string::npos);
  EXPECT_EQ(r.size(), 1u);
}

TEST(common_registry, an_unknown_name_is_not_found_and_lists_what_exists) {
  mqde::Registry<Animal> r;
  r.add("dog", [] { return std::make_unique<Dog>(); });
  r.add("ant", [] { return std::make_unique<Dog>(); });
  auto e = r.create("cat");
  ASSERT_FALSE(e);
  EXPECT_EQ(e.error().code, mqde::Errc::kNotFound);
  EXPECT_NE(e.error().message.find("available: ant, dog"), std::string::npos) << e.error().message;
  mqde::Registry<Animal> empty;
  EXPECT_NE(empty.create("x").error().message.find("(none)"), std::string::npos);
}

TEST(common_registry, bad_arguments_are_rejected) {
  mqde::Registry<Animal> r;
  EXPECT_FALSE(r.add("", [] { return std::make_unique<Dog>(); }));
  EXPECT_FALSE(r.add("x", nullptr));
  EXPECT_EQ(r.size(), 0u);
}

TEST(common_registry, concurrent_adds_of_one_name_let_exactly_one_win) {
  mqde::Registry<Animal> r;
  std::atomic<int> won{0};
  std::vector<std::thread> ts;
  for (int i = 0; i < 8; ++i)
    ts.emplace_back([&] {
      if (r.add("dog", [] { return std::make_unique<Dog>(); })) ++won;
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(won.load(), 1);
}

TEST(common_registry, the_registrar_reports_a_duplicate_instead_of_hiding_it) {
  mqde::Registry<Animal> r;
  mqde::Registrar<Animal, Dog> a(r, "dog"), b(r, "dog");
  EXPECT_TRUE(a.ok);
  EXPECT_FALSE(b.ok);
  EXPECT_NE(b.error.find("already registered"), std::string::npos);
}
