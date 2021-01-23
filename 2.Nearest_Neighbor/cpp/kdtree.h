//
// Created by xslittlegrass on 4/19/20.
//

#pragma once

#include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stack>
#include <vector>
#include "Eigen/Core"

template <typename T>
class Kdtree 
{
  
  using Point = Eigen::Matrix<T, 3, 1>;

  public:
  struct Node 
  {
    T value_;
    uint32_t axis_;
    Node* left_;
    Node* right_;
    uint32_t point_index_begin_;
    uint32_t point_index_end_;
  };

  Kdtree(const std::vector<Point>& points, uint32_t leaf_size = 15)
      : points_(points), leaf_size_(leaf_size) 
      {
        root_ = build_tree(0, points_.size());
      }
  ~Kdtree() 
      {   
        points_.clear(); 
      }

  bool query(const Point& point, T min_squared_dist, Point* nearest_point) 
  {
    uint32_t nearest_point_index = std::numeric_limits<uint32_t>::max();

    search(root_, point, &min_squared_dist, &nearest_point_index);

    if (nearest_point_index != std::numeric_limits<uint32_t>::max()) 
    {
      *nearest_point = points_[nearest_point_index];
      return true;
    } 
    else 
    {
      return false;
    }
  }

  void print() 
  { 
    std::cout << to_string(root_) << std::endl; 
  }

  void serialize
  (
      std::vector<uint8_t>* axis_list,
      std::vector<T>* value_list,
      std::vector<uint32_t>* start_end_indices) {
    serialization(axis_list, value_list, start_end_indices);
  }

  void serialize_to_file(const std::string& file_path) 
  {
    std::vector<uint8_t> axis_list;
    std::vector<T> value_list;
    std::vector<uint32_t> start_end_indices;

    serialization(&axis_list, &value_list, &start_end_indices);
    {
      std::ofstream file(
          file_path + ".points", std::ios::out | std::ios::binary);
      for (const Point& point : points_) {
        file << point[0] << point[1] << point[2];
      }
      file.close();
    }
    {
      std::ofstream file(file_path + ".axis", std::ios::out | std::ios::binary);
      for (uint8_t axis : axis_list) {
        file << axis;
      }
      file.close();
    }
    {
      std::ofstream file(
          file_path + ".indices", std::ios::out | std::ios::binary);
      for (uint32_t index : start_end_indices) {
        file << index;
      }
      file.close();
    }
    {
      std::ofstream file(
          file_path + ".values", std::ios::out | std::ios::binary);
      for (T value : value_list) {
        file << value;
      }
      file.close();
    }
  }

 private:
  Node* build_tree(uint32_t begin, uint32_t end, uint32_t depth = 0) 
  {
    if (begin == end) 
    {
      return nullptr;
    }

    // create leaf node
    if (end - begin <= leaf_size_) {
      Node* node = new Node();
      node->point_index_begin_ = begin;
      node->point_index_end_ = end;
      node->left_ = node->right_ = nullptr;
      return node;
    }

    // current splitting axis
    auto min_max_x = std::minmax_element(
        points_.begin() + begin,
        points_.begin() + end,
        [](const Point& a, const Point& b) { return a[0] < b[0]; });
    auto min_max_y = std::minmax_element(
        points_.begin() + begin,
        points_.begin() + end,
        [](const Point& a, const Point& b) { return a[1] < b[1]; });
    auto min_max_z = std::minmax_element(
        points_.begin() + begin,
        points_.begin() + end,
        [](const Point& a, const Point& b) { return a[2] < b[2]; });
    T dx = (*min_max_x.second)[0] - (*min_max_x.first)[0];
    T dy = (*min_max_y.second)[1] - (*min_max_y.first)[1];
    T dz = (*min_max_z.second)[2] - (*min_max_z.first)[2];
    uint32_t axis = (dx > dy ? (dx > dz ? 0 : 2) : (dy > dz ? 1 : 2)); //把范围最大的作为axis，如：dx如果最大，则axis=0
    //  // uint32_t axis = depth % 3;
    // only need to find the median and separate data by the median, this is
    // O(N) vs O(N log(N)) for sorting
    std::nth_element(
        points_.begin() + begin,
        points_.begin() + (begin + end) / 2,
        points_.begin() + end,
        [axis](const Point& a, const Point& b) { return a[axis] < b[axis]; });

    Node* node = new Node();
    node->axis_ = axis;
    uint32_t median = (begin + end) / 2;
    node->value_ = points_[median][axis];

    node->left_ = build_tree(begin, median, depth + 1);
    node->right_ = build_tree(median, end, depth + 1);

    return node;
  }

  std::string to_string(Node* node) {
    if (!node) {
      return "";
    }
    std::stringstream buffer;
    buffer << "[(" << node->axis_ << "," << node->value_ << ")"
           << ",[" << to_string(node->left_) << "],[" << to_string(node->right_)
           << "]]";
    return std::string(buffer.str());
  }

  void search(Node* node, const Point& point, T* min_dist, uint32_t* best_point_index) 
  {
    if (!node) 
    {
      return;
    }

    // look at leaf node
    if (node->left_ == nullptr && node->right_ == nullptr) 
    {
      auto best_point_iter = std::min_element(
          points_.begin() + node->point_index_begin_,
          points_.begin() + node->point_index_end_,
          [&point](Point& p1, Point& p2) 
          {
            return (p1 - point).squaredNorm() < (p2 - point).squaredNorm();
          });
      T best_dist = (*best_point_iter - point).squaredNorm();
      if (best_dist < *min_dist) 
      {
        *min_dist = best_dist;
        *best_point_index = best_point_iter - points_.begin();
      }
      return;
    }

    // if point fall into left branch, check left branch
    if (point[node->axis_] < node->value_) 
    {
      search(node->left_, point, min_dist, best_point_index);
      // if the search distance still covers the right branch, look at the right
      // branch
      T dist_to_boundary = point[node->axis_] - node->value_;
      if (dist_to_boundary * dist_to_boundary < *min_dist) 
      {
        search(node->right_, point, min_dist, best_point_index);
      }
    }
    // the same if the point falls into the right branch
    else 
    {
      search(node->right_, point, min_dist, best_point_index);
      T dist_to_boundary = point[node->axis_] - node->value_;
      if (dist_to_boundary * dist_to_boundary < *min_dist) 
      {
        search(node->left_, point, min_dist, best_point_index);
      }
    }
  }

  void serialization(
      std::vector<uint8_t>* axis_list,
      std::vector<T>* value_list,
      std::vector<uint32_t>* start_end_indices) 
  {
    // seralize by level

    axis_list->clear();
    value_list->clear();
    start_end_indices->clear();

    axis_list->reserve(points_.size());
    value_list->reserve(points_.size());
    start_end_indices->reserve(points_.size());

    std::stack<Node*> node_stack;
    node_stack.push(root_);

    while (!node_stack.empty()) 
    {
      Node* node = node_stack.top();
      node_stack.pop();

      if (node->left_ || node->right_) 
      {
        // internal node
        if (node->left_) 
        {
          node_stack.push(node->left_);
        }
        if (node->right_) 
        {
          node_stack.push(node->right_);
        }
        axis_list->push_back((uint8_t)node->axis_);
        value_list->push_back(node->value_);
      } 
      else 
      {
        // leaf node
        start_end_indices->push_back(node->point_index_begin_);
        start_end_indices->push_back(node->point_index_end_);
      }
    }
  }

  Node* root_;
  std::vector<Point> points_;
  uint32_t leaf_size_;
};

