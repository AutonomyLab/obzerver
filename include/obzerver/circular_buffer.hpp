#ifndef CIRCULAR_BUFFER
#define CIRCULAR_BUFFER

#include <deque>
#include <cassert>
#include <stdexcept>

template<typename T>
class CircularBuffer {
private:
  std::size_t max_sz;
  std::deque<T> buffer;
  T dummy;

public:
  typedef typename std::deque<T>::iterator iterator;
  typedef typename std::deque<T>::const_iterator const_iterator;
  typedef typename std::deque<T>::reverse_iterator reverse_iterator;
  typedef typename std::deque<T>::const_reverse_iterator const_reverse_iterator;
  typedef typename std::deque<T>::reference reference;
  typedef typename std::deque<T>::const_reference const_reference;

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

  void push_back(const  T& val) {
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

  reference front() {
    return buffer.front();
  }

  const_reference front() const {
    return buffer.front();
  }

  reference back() {
    return buffer.back();
  }

  const_reference back() const {
    return buffer.back();
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

  reverse_iterator rbegin() {
    return buffer.rbegin();
  }

  const_reverse_iterator rbegin() const {
    return buffer.rbegin();
  }

  reverse_iterator rend() {
    return buffer.rend();
  }

  const_reverse_iterator rend() const {
    return buffer.rend();
  }

  // These are two special functions
  // if index i corresponds to time t-i
  // prev(offset) is object at time t-offset
  // You should always use push_front() to benefit from this
  // throws std::out_of_range on failure
  const_reference prev(const std::size_t offset = 1) const {
    return buffer.at(offset);
  }

  const_reference latest() const {
    if (!buffer.size()) {
        throw std::out_of_range("Cannot access latest, Circular Buffer is empty.");
    }
    return buffer.front();
  }
};

#endif
