//
// Created by xslittlegrass on 4/19/20.
//

#pragma once

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "Eigen/Core"

template <typename T>
class Octree {
  using Point = Eigen::Matrix<T, 3, 1>;

 public:
  struct Node {
    Point center_;
    T extent_;  // half size
    Node* child[8];
    uint32_t point_index_begin_;
    uint32_t point_index_end_;
  };

  Octree(
      const std::vector<Point>& points,
      const Point& center,
      T extent,
      uint32_t leaf_size = 15)
      : points_(points), leaf_size_(leaf_size) {
    root_ = build_tree(0, points_.size(), center, extent);
  }

  void print() { std::cout << to_string(root_) << std::endl; }

  bool query(const Point& point, T min_squared_dist, Point* nearest_point) {
    uint32_t nearest_point_index = std::numeric_limits<uint32_t>::max();

    search(root_, point, &min_squared_dist, &nearest_point_index);

    if (nearest_point_index != std::numeric_limits<uint32_t>::max()) {
      *nearest_point = points_[nearest_point_index];
      return true;
    } else {
      return false;
    }
  }

 private:
  Node* build_tree(
      uint32_t point_index_begin,
      uint32_t point_index_end,
      const Point& center,
      T extent) {
    if (point_index_begin == point_index_end) {
      return nullptr;
    }

    // return if leaf node
    if (point_index_end - point_index_begin <= leaf_size_) {
      Node* node = new Node();
      node->point_index_begin_ = point_index_begin;
      node->point_index_end_ = point_index_end;
      node->center_ = center;
      node->extent_ = extent;
      for (int i = 0; i < 8; i++) {
        node->child[i] = nullptr;
      }
      return node;
    }

    // divide the octant and split the points

    std::vector<int> octant_indices;
    octant_indices.reserve(point_index_end - point_index_begin);

    std::vector<std::vector<Point>> points_ls(8);
    for (int i = 0; i < 8; i++) {
      points_ls[i].reserve((point_index_end - point_index_begin));
    }

    for (uint32_t i = point_index_begin; i < point_index_end; i++) {
      int index = (points_[i].z() > center.z()) << 2
          | (points_[i].y() > center.y()) << 1 | (points_[i].x() > center.x());
      points_ls[index].push_back(points_[i]);
    }

    std::vector<int> child_node_size(8);
    int counts = 0;
    for (int i = 0; i < 8; i++) {
      std::move(
          points_ls[i].begin(),
          points_ls[i].end(),
          points_.begin() + point_index_begin + counts);
      child_node_size[i] = points_ls[i].size();
      counts += points_ls[i].size();
    }

    // create child node

    Node* node = new Node();
    node->center_ = center;
    node->extent_ = extent;

    // child node center
    T child_extend = node->extent_ / 2;
    std::vector<Point> child_center(8);
    Point child_min = center.array() - child_extend;
    Point child_max = center.array() + child_extend;
    child_center[0] = Point(child_min.x(), child_min.y(), child_min.z());
    child_center[1] = Point(child_max.x(), child_min.y(), child_min.z());
    child_center[2] = Point(child_min.x(), child_max.y(), child_min.z());
    child_center[3] = Point(child_max.x(), child_max.y(), child_min.z());
    child_center[4] = Point(child_min.x(), child_min.y(), child_max.z());
    child_center[5] = Point(child_max.x(), child_min.y(), child_max.z());
    child_center[6] = Point(child_min.x(), child_max.y(), child_max.z());
    child_center[7] = Point(child_max.x(), child_max.y(), child_max.z());

    int counts2 = 0;
    for (int i = 0; i < 8; i++) {
      Point child_c;
      node->child[i] = build_tree(
          point_index_begin + counts2,
          point_index_begin + counts2 + child_node_size[i],
          child_center[i],
          child_extend);
      counts2 += child_node_size[i];
    }

    return node;
  }

  std::string to_string(Node* node) {
    if (!node) {
      return "";
    }
    std::stringstream buffer;
    buffer << "[("
           << "(" << node->center_.transpose() << ")"
           << ",(" << node->extent_ << "))";
    bool is_leaf = true;
    for (int i = 0; i < 8; i++) {
      if (node->child[i]) {
        is_leaf = false;
        buffer << "," << to_string(node->child[i]);
      }
    }
    if (is_leaf) {
      for (int i = node->point_index_begin_; i < node->point_index_end_; i++) {
        buffer << ",(" << points_[i].transpose() << ")";
      }
    }
    buffer << "]";
    return std::string(buffer.str());
  }

  void search(
      Node* node, const Point& point, T* min_dist, uint32_t* best_point_index) {
    bool is_leaf = true;
    for (int i = 0; i < 8; i++) {
      if (node->child[i]) {
        is_leaf = false;
      }
    }
    // look at leaf node
    if (is_leaf) {
      auto best_point_iter = std::min_element(
          points_.begin() + node->point_index_begin_,
          points_.begin() + node->point_index_end_,
          [&point](Point& p1, Point& p2) {
            return (p1 - point).squaredNorm() < (p2 - point).squaredNorm();
          });
      T best_dist = (*best_point_iter - point).squaredNorm();
      if (best_dist < *min_dist) {
        *min_dist = best_dist;
        *best_point_index = best_point_iter - points_.begin();
      }
      return;
    }

    // determine the octant that the query point fall into
    int index = (point.z() > node->center_.z()) << 2
        | (point.y() > node->center_.y()) << 1
        | (point.x() > node->center_.x());

    // first look at the octant the point falls into
    //    std::cout << "look at query point octant: " << index << std::endl;
    if (node->child[index]) {
      search(node->child[index], point, min_dist, best_point_index);
    }
    // then look at the octant by the oder
    for (uint8_t idx : query_oder[index]) {
      if (node->child[idx]) {
        if (overlaps(node->child[idx], point, *min_dist)) {
          search(node->child[idx], point, min_dist, best_point_index);
        }
      }
    }
  }

  bool overlaps(const Node* node, const Point& query, T sqRadius) {
    Point diff = (query - node->center_).cwiseAbs();

    T radius = sqrt(sqRadius);

    float maxdist = radius + node->extent_;

    // return false if the point is out side of the relavant region
    if (diff.x() > maxdist || diff.y() > maxdist || diff.z() > maxdist) {
      return false;
    }

    // within the planes of the octrant
    if (diff.x() < node->extent_ || diff.y() < node->extent_
        || diff.z() < node->extent_) {
      return true;
    }

    // finally check the corner
    return Point(diff.array() - node->extent_).cwiseMax(0).squaredNorm()
        < sqRadius;
  }

  Node* root_;
  std::vector<Point> points_;
  uint32_t leaf_size_;

  const std::vector<std::vector<uint8_t>> query_oder = {{1, 2, 4, 3, 5, 6, 7},
                                                        {0, 3, 5, 2, 4, 7, 6},
                                                        {0, 3, 6, 1, 4, 7, 5},
                                                        {1, 2, 7, 0, 5, 6, 4},
                                                        {0, 5, 6, 1, 2, 7, 3},
                                                        {1, 4, 7, 0, 3, 6, 2},
                                                        {2, 4, 7, 0, 3, 5, 1},
                                                        {3, 5, 6, 1, 2, 4, 0}};
};
