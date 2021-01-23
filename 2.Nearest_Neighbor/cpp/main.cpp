//
// Created by xslittlegrass on 4/19/20.
//

#include "kdtree.h"
#include "octree.h"
#include <iostream>
#include <fstream>
#include <chrono>

void ReadPointCloud(const std::string& path, std::vector<Eigen::Vector3f>* points)
{
  points->clear();
  points->reserve(100000);
  std::ifstream file(path,std::ios::binary);
  float x,y,z,i;
  while(file)
  {
    file.read((char*)&x,sizeof(float));
    file.read((char*)&y,sizeof(float));
    file.read((char*)&z,sizeof(float));
    file.read((char*)&i,sizeof(float));
    points->emplace_back(x,y,z);
  }
}

int main(int argc, char** argv){

  std::string point_cloud_dir = "/media/xslittlegrass/0EC37D6E1F9BDD00/kitti/sequences/00/velodyne/";

  std::chrono::duration<double> kdtree_build(0);
  std::chrono::duration<double> kdtree_query(0);
  std::chrono::duration<double> octree_build(0);
  std::chrono::duration<double> octree_query(0);

  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;

  for(int i=0;i<99;i++){
    std::string file_name = std::to_string(i);
    file_name = std::string(6-file_name.size(), '0') + file_name;
    std::string file_path = point_cloud_dir + file_name + ".bin";
    std::vector<Eigen::Vector3f> points;
    ReadPointCloud(file_path,&points);

    Eigen::Vector3f query_point = points[0];
    float min_squared_dist = 1;

    t1 = std::chrono::high_resolution_clock::now();

    Kdtree<float> kdtree(points);

    t2 = std::chrono::high_resolution_clock::now();

    kdtree_build += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    t1 = std::chrono::high_resolution_clock::now();

    Eigen::Vector3f nearest_point;
    kdtree.query(query_point,min_squared_dist,&nearest_point);

    t2 = std::chrono::high_resolution_clock::now();

    kdtree_query += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    t1 = std::chrono::high_resolution_clock::now();

    Octree<float> octree(points,Eigen::Vector3f::Zero(),200);

    t2 = std::chrono::high_resolution_clock::now();

    octree_build += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    t1 = std::chrono::high_resolution_clock::now();

    octree.query(query_point,min_squared_dist,&nearest_point);

    t2 = std::chrono::high_resolution_clock::now();

    octree_query += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

  }

  std::cout << "kdtree build: " << kdtree_build.count() << " s." << std::endl;
  std::cout << "kdtree query: " << kdtree_query.count() << " s." << std::endl;
  std::cout << "octree build: " << octree_build.count() << " s." << std::endl;
  std::cout << "octree query: " << octree_query.count() << " s." << std::endl;

  return 0;
}