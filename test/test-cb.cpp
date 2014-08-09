#include "gtest/gtest.h"
#include "obzerver/circular_buffer.hpp"

struct dummy_t {
  double x;
  double y;
};

class CBTest : public ::testing::Test {
protected:

  CircularBuffer<int> cb1;
  CircularBuffer<double> cb2;

  CBTest(): cb1(100), cb2(50, 3.14) {;}
};

TEST_F(CBTest, TestSize) {
  ASSERT_EQ(cb1.size(), (std::size_t) 0);
  ASSERT_EQ(cb1.max_size(), (std::size_t) 100);
  ASSERT_EQ(cb2.size(), (std::size_t) 50);
  ASSERT_EQ(cb2.max_size(), (std::size_t) 50);
}

TEST_F(CBTest, TestPushPop) {
  cb1.push_front(10);
  ASSERT_EQ(cb1.size(), (std::size_t) 1);
  ASSERT_EQ(cb1.at(0), 10);
  ASSERT_EQ(cb1[0], 10);

  cb1.pop_front();

  ASSERT_EQ(cb1.size(), (std::size_t) 0);

  for (std::size_t i = 0; i < 1000; i++) {
      ASSERT_EQ(cb1.size(), std::min(i, (std::size_t) 100));
      cb1.push_back(i);
      cb2.push_back(static_cast<double>(i));
  }
  ASSERT_EQ(cb1.size(), (std::size_t) 100);
  ASSERT_EQ(cb2.size(), (std::size_t) 50);

  cb2.clear();
  cb2.push_back(1.0); // 1.0
  cb2.push_front(2.0); // 2.0 1.0

  ASSERT_EQ(cb2.size(), (std::size_t) 2);
  ASSERT_EQ(cb2[0], 2.0);
  ASSERT_EQ(cb2[1], 1.0);
}

TEST_F(CBTest, TestIterator) {
  cb1.clear();

  for (std::size_t i = 0; i < 100; i++) {
    cb1.push_back(i % 2);
  }

  ASSERT_EQ(cb1.size(), (std::size_t) 100);

  std::size_t i = 0;
  for (CircularBuffer<int>::iterator it = cb1.begin(); it != cb1.end(); it++) {
      ASSERT_EQ(*it, (int) i++ % 2);
  }
}

TEST_F(CBTest, FrontBack) {
  cb1.clear();
  cb1.resize(2, 10); // 10 10
  ASSERT_EQ(cb1.size(), (std::size_t) 2);
  ASSERT_EQ(cb1.front(), 10);
  ASSERT_EQ(cb1.back(), cb1.front());

  cb1.push_front(20); // 20 10
  ASSERT_EQ(cb1.size(), (std::size_t) 2);
  ASSERT_EQ(cb1.front(), 20);
  ASSERT_EQ(cb1.back(), 10);

  cb1.clear();
  cb1.push_front(1000); // 1000
  cb1.push_front(2000); // 2000 1000

  ASSERT_EQ(cb1.front(), 2000);
  ASSERT_EQ(cb1.latest(), cb1.front());
  ASSERT_EQ(cb1.prev(), 1000);
}

TEST_F(CBTest, LatestPrev) {
  cb1.resize(2, 0);
  cb1.clear();

  ASSERT_NO_THROW(cb1.front()); // This is why using latest() is a better choice
  ASSERT_THROW(cb1.latest(), std::out_of_range);
  cb1.push_front(15);
  ASSERT_NO_THROW(cb1.latest());
  ASSERT_THROW(cb1.prev(), std::out_of_range);

  cb1.push_front(25);
  ASSERT_EQ(cb1.latest(), 25);
  ASSERT_EQ(cb1.prev(), 15);

  cb1.push_front(35);
  ASSERT_EQ(cb1.latest(), 35);
  ASSERT_EQ(cb1.prev(), 25);
  ASSERT_EQ(cb1.size(), (std::size_t) 2);
}
