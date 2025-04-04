#ifndef BUNDLEFIT_CONSTEXPR_DICT_H
#define BUNDLEFIT_CONSTEXPR_DICT_H

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace bundlefit {
// const expr dictionary
template <typename Key, typename Value, uint32_t Size>
class _const_dict_impl {
private:
  // static_assert(static_cast<uint32_t>(Key::_num_items) == Size, "loader type
  // and readable are mismatched");
  using Pair = std::pair<Key, Value>;
  std::array<Pair, Size> dict_;

protected:
  constexpr explicit _const_dict_impl(const Pair (&arr)[Size]) noexcept
      : dict_() {
    for (uint32_t i = 0; i < Size; i++) {
      dict_[i].first  = arr[i].first;
      dict_[i].second = arr[i].second;
    }
  }

public:
  template <typename K, typename V, size_t S>
  friend constexpr auto make_const_dict(const std::pair<K, V> (&arr)[S]);

  constexpr auto operator[](Key key) const noexcept -> Value {
    for (uint32_t i = 0; i < Size; i++) {
      if (dict_[i].first == key) {
        return dict_[i].second;
      }
    }
    assert(false && "Invalid item access to const table");
    return {};
  }
  constexpr auto get_key(const std::string& s) const noexcept
      -> const std::optional<Key> {
    for (uint32_t i = 0; i < Size; i++) {
      if (dict_[i].second == s) {
        return dict_[i].first;
      }
    }
    return std::nullopt;
  }
  constexpr auto max() const noexcept -> Value {
    Value max_ = dict_[0].second;
    for (uint32_t i = 1; i < Size; i++) {
      if (max_ < dict_[i].second) {
        max_ = dict_[i].second;
      }
    }
    return max_;
  }
};

template <typename Key, typename Value = std::string_view, size_t Size>
constexpr auto make_const_dict(const std::pair<Key, Value> (&arr)[Size]) {
  return _const_dict_impl<Key, Value, Size>{arr};
}
} // namespace bundlefit
#endif // BUNDLEFIT_CONSTEXPR_DICT_H
