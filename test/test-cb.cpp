#include "gtest/gtest.h"
#include "obzerver/circular_buffer.h"

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
  ASSERT_EQ(cb1.size(), 0);
  ASSERT_EQ(cb1.max_size(), 100);
  ASSERT_EQ(cb2.size(), 50);
  ASSERT_EQ(cb2.max_size(), 50);
}

TEST_F(CBTest, TestPushPop) {
  cb1.push_front(10);
  ASSERT_EQ(cb1.size(), 1);
  ASSERT_EQ(cb1.at(0), 10);
  ASSERT_EQ(cb1[0], 10);

  cb1.pop_front();

  ASSERT_EQ(cb1.size(), 0);

  for (unsigned int i = 0; i < 1000; i++) {
      ASSERT_EQ(cb1.size(), std::min((int)i, 100));
      cb1.push_back(i);
      cb2.push_back(static_cast<double>(i));
  }
  ASSERT_EQ(cb1.size(), 100);
  ASSERT_EQ(cb2.size(), 50);

  cb2.clear();
  cb2.push_back(1.0); // 1.0
  cb2.push_front(2.0); // 2.0 1.0

  ASSERT_EQ(cb2.size(), 2);
  ASSERT_EQ(cb2[0], 2.0);
  ASSERT_EQ(cb2[1], 1.0);
}

TEST_F(CBTest, TestIterator) {
  cb1.clear();

  for (unsigned int i = 0; i < 100; i++) {
    cb1.push_back(i % 2);
  }

  ASSERT_EQ(cb1.size(), 100);

  unsigned int i = 0;
  for (CircularBuffer<int>::iterator it = cb1.begin(); it != cb1.end(); it++) {
      ASSERT_EQ(*it, i++ % 2);
  }
}
