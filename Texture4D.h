#pragma once

#include "RayLight.h"
#include "CudaPointers.h"

#include<stdio.h>
#include<opencv2/opencv.hpp>
#include <fstream>

template <class T = double>
class Texture4D
{
public:
	cv::Mat texture;

	size_t usize, vsize, ssize, tsize, totalSize;

	T* d_out = 0;
	int* d_size = 0;

	Texture4D(size_t usize, size_t vsize, size_t ssize, size_t tsize) : usize(usize), vsize(vsize), ssize(ssize), tsize(tsize), totalSize(usize* vsize* ssize* tsize) {
		texture = cv::Mat::zeros(totalSize, 1, CV_64FC4);
	}
	~Texture4D() {
		texture.release();
	}

	inline size_t getCoord(const size_t u, const size_t v, const size_t s, const size_t t) {
		return tsize * (ssize * (vsize * u + v) + s) + t;
	}
	inline size_t getCoord(cv::Vec<size_t, 4> coord) {
		return tsize * (ssize * (vsize * coord[0] + coord[1]) + coord[2]) + coord[3];
	}
	inline size_t getCoord(size_t* coord) {
		return tsize * (ssize * (vsize * coord[0] + coord[1]) + coord[2]) + coord[3];
	}

	inline cv::Vec<T,4>& operator()(const int u, const int v, const int s, const int t) {
		return texture.at<cv::Vec<T, 4>>(getCoord(u, v, s, t));
	};

	void compileToUnity(std::string s);

	void RayLightGeneratorCuda(int v, int s, int t, T radius, int ws, int wsTotal, CudaPointers<T>& cp);
};

