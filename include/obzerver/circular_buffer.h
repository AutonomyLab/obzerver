#ifndef CIRCULAR_BUFFER
#define CIRCULAR_BUFFER

#include <deque>
#include <cassert>

template<typename T>
class CircularBuffer {
private:
  std::size_t max_sz;
  std::deque<T> buffer;


public:
  typedef typename std::deque<T>::iterator iterator;
  typedef typename std::deque<T>::const_iterator const_iterator;

  // This constructor does not fill the buffer, so the buffer size is 0
  CircularBuffer(const std::size_t max_size): max_sz(max_size) {;}

  // This fills the buffer with fill, so the buffer size will be sz
  CircularBuffer(const std::size_t sz, const T& fill): max_sz(sz), buffer(sz, fill) {;}

  const CircularBuffer& operator=(const CircularBuffer& rhs) {
    max_sz = rhs.max_sz;
    buffer = rhs.buffer;
    return *this;
  }

  void push_front(const T& val) {
    if (buffer.size() == max_sz) buffer.pop_back();
    buffer.push_front(val);
    assert(buffer.size() <= max_sz);
  }

  void push_back(const T& val) {
    if (buffer.size() == max_sz) buffer.pop_front();
    buffer.push_back(val);
    assert(buffer.size() <= max_sz);
  }

  void pop_front() {
    buffer.pop_front();
  }

  void pop_back() {
    buffer.pop_back();
  }

  T& at(std::size_t index) {
    return buffer.at(index);
  }

  const T& at(std::size_t index) const {
    return buffer.at(index);
  }

  T& operator[](std::size_t index) {
    return buffer[index];
  }

  const T& operator[](std::size_t index) const {
    return buffer[index];
  }

  bool empty() const {
    return (buffer.size() == 0);
  }

  std::size_t size() const {
    return buffer.size();
  }

  std::size_t max_size() const {
    return max_sz;
  }

  void resize(const std::size_t sz, const T& val) {
    max_sz = sz;
    buffer.resize(sz, val);
  }

  void clear() {
    buffer.clear();
  }

  iterator begin() {
    return buffer.begin();
  }

  const_iterator begin() const {
    return buffer.begin();
  }

  iterator end() {
    return buffer.end();
  }

  const_iterator end() const {
    return buffer.end();
  }

  iterator rbegin() {
    return buffer.rbegin();
  }

  const_iterator rbegin() const {
    return buffer.rbegin();
  }

  iterator rend() {
    return buffer.rend();
  }

  const_iterator rend() const {
    return buffer.rend();
  }

};

#endif
