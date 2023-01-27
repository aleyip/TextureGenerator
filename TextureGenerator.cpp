// TextureGenerator.cpp : Este arquivo contém a função 'main'. A execução do programa começa e termina ali.
//

#include <math.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include "defines.h"

#include "Object.h"
#include "Sphere.h"
#include "Cylinder.h"
#include "Cube.h"

#include "RayLight.h"
#include "texture4D.h"

#include "CudaPointers.h"

#include "Light.h"
#include "PointLight.h"
#include "DirectionalLight.h"

#define NUMOBJ 7
#define NUMLIGHT 1

#define WINDOW_SIZE 3
#define BUFFERSIZE 50000000

#define specPower 0.3
#define specShiny 16
#define ambientPower 0.1

#define U_SIZE 512
#define V_SIZE 256
#define S_SIZE 33
#define T_SIZE 33

using ObjectT = Object<typeT>;
using SphereT = Sphere<typeT>;
using CylinderT = Cylinder<typeT>;
using CubeT = Cube<typeT>;

using RayLightT = RayLight<typeT>;
using Texture4DT = Texture4D<typeT>;
using vec3T = vec3<typeT>;
using CudaPointersT = CudaPointers<typeT>;

using LightT = Light<typeT>;
using PointLightT = PointLight<typeT>;
using DirectionalLightT = DirectionalLight<typeT>;

void generateViewPort(ObjectT** obj, LightT** light, vec3T viewerPos, vec3T viewerDir, typeT fov, std::string nameFile, int width, int height, int aliasing) {
	//PointLight<typeT> light = PointLight<typeT>(vec3T(-6, 4.5, 2), cv::Vec<typeT, 3>(1, 1, 1));
	
	
	cv::Mat img = cv::Mat(width * aliasing, height * aliasing, CV_64FC4);
	typeT halfHeight = tan(fov * 0.01745329251994329576923690768489 / 2.);

	viewerDir.normalize();
	vec3T right = vec3(-viewerDir.z, 0., viewerDir.x);
	right.normalize();
	vec3T up = vec3(-right.z * viewerDir.y, right.z * viewerDir.x - right.x * viewerDir.z, right.x * viewerDir.y);
	up.normalize();

	printf("Viewer Position: %f, %f, %f\n", viewerPos.x, viewerPos.y, viewerPos.z);
	printf("Viewer Direction: %f, %f, %f\n", viewerDir.x, viewerDir.y, viewerDir.z);
	img.forEach<cv::Vec<typeT, 4>>([&obj, &light, &viewerPos, &viewerDir, img, halfHeight, &right, &up](cv::Vec<typeT, 4>& pixel, const int* pos)->void {
		typeT dV = ((1.0 - 2.0 * pos[0] / typeT(img.rows - 1)) * halfHeight);
		typeT dH = ((1.0 - 2.0 * pos[1] / typeT(img.cols - 1)) * halfHeight);

		vec3T vDaux = vec3T(1, dV, dH);
		vDaux.normalize();
		vec3T vD = vec3T(
			viewerDir.x * vDaux.x + up.x * vDaux.y + right.x * vDaux.z,
			viewerDir.y * vDaux.x + up.y * vDaux.y + right.y * vDaux.z,
			viewerDir.z * vDaux.x + up.z * vDaux.y + right.z * vDaux.z
		);

		RayLightT ray = RayLightT(viewerPos, vD);
		vec3T normal_out, normal, collision_out, collision;
		int objsel = -1;
		typeT dist = -1;
		for (int i = 0; i < NUMOBJ; i++)
		{
			typeT d = obj[i]->CheckCollision(ray, collision_out, normal_out);
			if ((d < dist && d >= 0) || dist == -1) {
				dist = d;
				normal = normal_out;
				collision = collision_out;
				objsel = i;
			}
		}

		if (dist >= 0) {
			pixel = Light<typeT>::setAmbientLight(*obj[objsel], ambientPower);
			for (int i = 0; i < NUMLIGHT; i++)
				pixel += light[i]->lightEffect(*obj[objsel], collision, normal, viewerPos);
		}
		else pixel = cv::Vec4d(0, 0, 0, 0);
	});

	cv::resize(img, img, cv::Size(width, height));
	img.convertTo(img, CV_8UC4, 255, 0);
	cv::imwrite(nameFile, img);
}

void generateViewPortCuda(ObjectT** obj, LightT** light, vec3T viewerPos, vec3T viewerDir, typeT fov, std::string nameFile, int width, int height, int aliasing) {

	cv::Mat img = cv::Mat(width * aliasing, height * aliasing, CV_64FC4);
	typeT halfHeight = tan(fov * 0.01745329251994329576923690768489 / 2.);

	viewerDir.normalize();
	vec3T right = vec3(-viewerDir.z, 0., viewerDir.x);
	right.normalize();
	vec3T up = vec3(-right.z * viewerDir.y, right.z * viewerDir.x - right.x * viewerDir.z, right.x * viewerDir.y);
	up.normalize();

	printf("Viewer Position: %f, %f, %f\n", viewerPos.x, viewerPos.y, viewerPos.z);
	printf("Viewer Direction: %f, %f, %f\n", viewerDir.x, viewerDir.y, viewerDir.z);

	CudaPointersT cp;
	cp.allocate(img.total(), 0, 0, 0, 0);
	{
		std::vector <cv::Vec<typeT, 4>> objColorList(NUMOBJ);
		std::vector <uint8_t> objShinyList(NUMOBJ);
		for (int i = 0; i < NUMOBJ; i++) {
			objColorList[i] = obj[i]->color;
			objShinyList[i] = obj[i]->specularShininness;
		}
		cp.uploadObjectColorProp(objColorList.data(), objShinyList.data(), NUMOBJ);
	}
	cp.setHitObjectList(-1, img.total());
	cp.setDistList(10e10, img.total());

	std::vector<RayLightT> rayList = std::vector<RayLightT>(img.total());
	for (int r = 0; r < img.rows; r++)
		for (int c = 0; c < img.cols; c++)
		{
			typeT dV = ((1.0 - 2.0 * r / typeT(img.rows - 1)) * halfHeight);
			typeT dH = ((1.0 - 2.0 * c / typeT(img.cols - 1)) * halfHeight);

			vec3T vDaux = vec3T(1, dV, dH);
			vDaux.normalize();
			vec3T vD = vec3T(
				viewerDir.x * vDaux.x + up.x * vDaux.y + right.x * vDaux.z,
				viewerDir.y * vDaux.x + up.y * vDaux.y + right.y * vDaux.z,
				viewerDir.z * vDaux.x + up.z * vDaux.y + right.z * vDaux.z
			);

			rayList[img.cols * r + c] = RayLightT(viewerPos, vD);
		}
	cp.uploadRayList(rayList.data(), img.total());

	for (int j = 0; j < NUMOBJ; j++)
		obj[j]->CheckCollisionCuda(cp, img.total(), j);

	Light<typeT>::setAmbientLightCUDA(cp, ambientPower, img.total());
	for (int j = 0; j < NUMLIGHT; j++)
		light[j]->addLightEffectsCUDA(cp, img.total());

	cp.downloadPixelColor((cv::Vec<typeT, 4> *)(img.data), img.total());
	cp.free();

	cv::resize(img, img, cv::Size(width, height));
	img.convertTo(img, CV_8UC4, 255, 0);
	cv::imwrite(nameFile, img);
}

void generateTexture(ObjectT** obj, LightT** light, std::string fileName, int usize, int vsize, int ssize, int tsize, int windowSize) {
	const int ws = windowSize * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	int size[] = { usize,vsize,ssize,tsize };
	Texture4DT tex = Texture4DT(size[0], size[1], size[2], size[3]);

	double inc[] = { M_PI * 2. / size[0], M_PI / (size[1] - 1.), M_PI / (size[2] - 1.), M_PI / (size[3] - 1.) };
	double halfinc[4];
	if (windowSize == 0) {
		halfinc[0] = 0;
		halfinc[1] = 0;
		halfinc[2] = 0;
		halfinc[3] = 0;
	}
	else {
		halfinc[0] = inc[0] / (2. * windowSize);
		halfinc[1] = inc[1] / (2. * windowSize);
		halfinc[2] = inc[2] / (2. * windowSize);
		halfinc[3] = inc[3] / (2. * windowSize);
	}

	typeT radius = 3.5;

	typeT denom = 1. / wsTotal;
	std::cout << denom << std::endl;

	std::cout << "Compile texture" << std::endl;
	auto start = std::chrono::steady_clock::now();
	for (int t = 0; t < size[3]; t++) {
		auto start = std::chrono::steady_clock::now();
		std::cout << "Step t" << t << std::endl;
		for (int s = 0; s < size[2]; s++) {
			std::cout << "Step t" << t << "/" << size[2] << " s " << s << "/" << size[3] << std::endl;
			auto start = std::chrono::steady_clock::now();
#pragma omp parallel for collapse(2)
			for (int v = 0; v < size[1]; v++) {
				for (int u = 0; u < size[0]; u++)
				{
					double angleBase[] = { -(inc[0] * u - M_PI), inc[1] * v - M_PI_2, inc[2] * s - M_PI_2, inc[3] * t - M_PI_2 };

					cv::Vec<typeT, 4> sumPixel = cv::Vec<typeT, 4>(0, 0, 0, 0);
					cv::Vec<typeT, 4> color;
					vec3T normal_out, normal, collision_out, collision;
					int objsel;
					typeT dist;
					double d;

					//#pragma omp parallel for collapse(4)
					for (int n = -windowSize; n <= windowSize; n++)
						for (int m = -windowSize; m <= windowSize; m++)
							for (int l = -windowSize; l <= windowSize; l++)
								for (int k = -windowSize; k <= windowSize; k++)
								{
									typeT angle[] = { angleBase[0] + n * halfinc[0], angleBase[1] + m * halfinc[1], angleBase[2] + k * halfinc[2], angleBase[3] + l * halfinc[3] };

									vec3T pos = vec3T(radius * sin(angle[0]) * cos(angle[1]), radius * sin(angle[1]), radius * cos(angle[0]) * cos(angle[1]));
									vec3T dir = vec3T(sin(angle[2]) * cos(angle[3]), sin(angle[3]), cos(angle[2]) * cos(angle[3]));

									vec3T versorForward(pos);
									versorForward.normalize();
									vec3T versorRight = vec3T(-versorForward.z, 0., versorForward.x);
									versorRight.normalize();
									vec3T versorUp = vec3T(-versorRight.z * versorForward.y, versorRight.z * versorForward.x - versorRight.x * versorForward.z, versorRight.x * versorForward.y);
									versorUp.normalize();

									vec3T vD = vec3T(
										versorRight.x * dir.x + versorUp.x * dir.y + versorForward.x * dir.z,
										versorRight.y * dir.x + versorUp.y * dir.y + versorForward.y * dir.z,
										versorRight.z * dir.x + versorUp.z * dir.y + versorForward.z * dir.z
									);
									vD = vD * (-1);
									RayLightT ray = RayLightT(pos, vD);
									vec3T collision, normal;

									//printf("%d %d %d %d %d %d %d %d angle: %f %f %f %f pos: %f %f %f dir: %f %f %f ", u, v, s, t, k, l, m, n, angle[0], angle[1], angle[2], angle[3], ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y, ray.direction.z);

									color = cv::Vec<typeT, 4>(0, 0, 0, 0);
									objsel = -1;
									dist = 10e10;
									for (int i = 0; i < NUMOBJ; i++)
									{
										d = obj[i]->CheckCollision(ray, collision_out, normal_out);
										if (d < dist && d >= 0) {
											dist = d;
											normal = normal_out;
											collision = collision_out;
											objsel = i;
										}
									}
									if (objsel != -1) {
										color = LightT::setAmbientLight(*obj[objsel], ambientPower);
										for (int i = 0; i < NUMLIGHT; i++)
											color += light[i]->lightEffect(*obj[objsel], collision, normal, ray.origin);
										sumPixel += color;
									}
								}
					tex(u, v, s, t) = sumPixel * denom;
				}
			}
			auto end = std::chrono::steady_clock::now();
			std::cout << "Elapsed time in milliseconds: "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
				<< " ms" << std::endl;
		}
		auto end = std::chrono::steady_clock::now();
		std::cout << "Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
	}

	cv::threshold(tex.texture, tex.texture, 1, 1, cv::THRESH_TRUNC);

	{
		auto start = std::chrono::steady_clock::now();
		tex.compileToImage(fileName);
		auto end = std::chrono::steady_clock::now();
		std::cout << "Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
	}
}

void generateTextureCuda(ObjectT** obj, LightT** light, std::string fileName, int usize, int vsize, int ssize, int tsize, int windowSize) {
	const int ws = windowSize * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	const int bufferSize = BUFFERSIZE / wsTotal;
	const int totalSize = usize * vsize * ssize * tsize;

	int size[] = { usize,vsize,ssize,tsize };
	Texture4DT tex = Texture4DT(size[0], size[1], size[2], size[3]);

	double inc[] = { M_PI * 2. / size[0], M_PI / (size[1] - 1.), M_PI / (size[2] - 1.), M_PI / (size[3] - 1.) };

	typeT radius = 3.5;

	typeT denom = 1. / wsTotal;
	std::cout << "Denominador: " << denom << std::endl;

	CudaPointersT cp;
	cp.allocate(bufferSize * wsTotal, usize, vsize, ssize, tsize);
	{
		std::vector <cv::Vec<typeT, 4>> objColorList(NUMOBJ);
		std::vector <uint8_t> objShinyList(NUMOBJ);
		for (int i = 0; i < NUMOBJ; i++) {
			objColorList[i] = obj[i]->color;
			objShinyList[i] = obj[i]->specularShininness;
		}
		cp.uploadObjectColorProp(objColorList.data(), objShinyList.data(), NUMOBJ);
	}
	std::cout << "Allocate: " << bufferSize * wsTotal << std::endl;

	std::cout << "Compile texture" << std::endl;
	int countLoop = 0;//size[3] * (size[2] * (size[1] * 0 + 0) + 16) + 16;
	int length = bufferSize;
	if (totalSize - countLoop < bufferSize)
		length = totalSize - countLoop;
	int bufferTotal;
	cv::Vec<typeT, 4>* dataPixel = (cv::Vec<typeT, 4>*)tex.texture.data;

	while (countLoop < totalSize)
	{
		auto start = std::chrono::steady_clock::now();

		bufferTotal = length * wsTotal;

		std::cout << countLoop << "/" << totalSize << std::endl;

		tex.RayLightGeneratorCuda(countLoop, length, radius, ws, wsTotal, cp);

		cp.setHitObjectList(-1, bufferTotal);
		cp.setDistList(10e10, bufferTotal);
		for (int o = 0; o < NUMOBJ; o++)
		{
			obj[o]->CheckCollisionCuda(cp, bufferTotal, o);
		}
		LightT::setAmbientLightCUDA(cp, 0.1, bufferTotal);
		for (int j = 0; j < NUMLIGHT; j++)
			light[j]->addLightEffectsCUDA(cp, bufferTotal);

		cp.pixelReduction(wsTotal, length);
		cp.downloadPixelColor(dataPixel, length);

		auto end = std::chrono::steady_clock::now();
		std::cout << " Compare Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
		//return;
		dataPixel += length;
		countLoop += length;
		if (totalSize - countLoop < bufferSize)
			length = totalSize - countLoop;
	} 
	//delete& dist;
	//delete& objSel;
	//delete& d;

	cp.free();

	{
		auto start = std::chrono::steady_clock::now();
		//tex.compileToUnity("D:\\testeGeneratorCuda.asset");
		tex.compileToImage(fileName);
		auto end = std::chrono::steady_clock::now();
		std::cout << "Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
	}
}

void generateMaps(ObjectT** obj, std::string fileName, int usize, int vsize, int windowSize) {
	const int ws = windowSize * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	const int totalSize = usize * vsize;

	int size[] = { usize,vsize };

	//Texture4DT tex = Texture4DT(size[0], size[1], size[2], size[3]);
	cv::Mat texture = cv::Mat(usize, vsize, CV_64FC4);
	cv::Mat normal = cv::Mat(usize, vsize, CV_64FC3);
	cv::Mat height = cv::Mat(usize, vsize, CV_64FC1);

	double inc[] = { M_PI * 2. / size[0], M_PI / (size[1] - 1.), M_PI / (size[2] - 1.), M_PI / (size[3] - 1.) };
	double halfinc[4];
	if (windowSize == 0) {
		halfinc[0] = 0;
		halfinc[1] = 0;
		halfinc[2] = 0;
		halfinc[3] = 0;
	}
	else {
		halfinc[0] = inc[0] / (2. * windowSize);
		halfinc[1] = inc[1] / (2. * windowSize);
		halfinc[2] = inc[2] / (2. * windowSize);
		halfinc[3] = inc[3] / (2. * windowSize);
	}

	typeT radius = 3.5;

	typeT denom = 1. / wsTotal;
	std::cout << denom << std::endl;

	std::cout << "Compile Map" << std::endl;
#pragma omp parallel for collapse(2)
	for (int v = 0; v < size[1]; v++) {
		for (int u = 0; u < size[0]; u++)
		{
			double angleBase[] = { -(inc[0] * u - M_PI), inc[1] * v - M_PI_2 };

			cv::Vec<typeT, 4> sumPixel = cv::Vec<typeT, 4>(0, 0, 0, 0);
			vec3T sumNormal = vec3T(0, 0, 0);
			typeT sumDist = 0;
			cv::Vec<typeT, 4> color;
			vec3T normal_out, normal_vec, collision_out, collision;
			int objsel;
			typeT dist;
			double d;

			//#pragma omp parallel for collapse(4)
			for (int n = -windowSize; n <= windowSize; n++)
				for (int m = -windowSize; m <= windowSize; m++)
					for (int l = -windowSize; l <= windowSize; l++)
						for (int k = -windowSize; k <= windowSize; k++)
						{
							typeT angle[] = { angleBase[0] + n * halfinc[0], angleBase[1] + m * halfinc[1] };

							vec3T pos = vec3T(radius * sin(angle[0]) * cos(angle[1]), radius * sin(angle[1]), radius * cos(angle[0]) * cos(angle[1]));

							vec3T vD(pos);
							vD.normalize();
							vD = vD * (-1);
							RayLightT ray = RayLightT(pos, vD);
							vec3T collision, normal;

							//printf("%d %d %d %d %d %d %d %d angle: %f %f %f %f pos: %f %f %f dir: %f %f %f ", u, v, s, t, k, l, m, n, angle[0], angle[1], angle[2], angle[3], ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y, ray.direction.z);

							color = cv::Vec<typeT, 4>(0, 0, 0, 0);
							objsel = -1;
							dist = 10e10;
							for (int i = 0; i < NUMOBJ; i++)
							{
								d = obj[i]->CheckCollision(ray, collision_out, normal_out);
								if (d < dist && d >= 0) {
									dist = d;
									normal_vec = normal_out;
									objsel = i;
								}
							}
							if (objsel != -1) {
								color = LightT::setAmbientLight(*obj[objsel], 1);
								sumPixel += color;
								sumNormal += normal_vec;
								sumDist += dist;
							}
						}
			texture.at<cv::Vec<typeT, 4>>(u, v) = sumPixel * denom;
			sumNormal.normalize();
			normal.at<cv::Vec<typeT, 3>>(u, v) = cv::Vec<typeT, 3>(normal_vec.x, normal_vec.y, normal_vec.z);
			height.at<typeT>(u, v) = sumDist * denom;
		}
	}

	std::string rootName = fileName.substr(0, rootName.length() - 4);
	std::string extension = fileName.substr(rootName.length() - 4, 4);

	cv::Mat img;
	texture.convertTo(img, CV_8UC4, 255);
	cv::transpose(img, img);
	//cv::imwrite("D:/testeTexMap.png", img);
	cv::imwrite(fileName, img);

	normal.convertTo(img, CV_8UC3, 127, 127);
	cv::transpose(img, img);
	cv::imwrite(rootName + "Normal" + extension, img);
	//cv::imwrite("D:/testeNormalMap.png", img);

	height.convertTo(img, CV_8UC1, -127 / (2 * radius), 127);
	cv::transpose(img, img);
	cv::imwrite(rootName + "Height" + extension, img);
	//cv::imwrite("D:/testeHeightMap.png", img);
}

void generateMapsCuda(ObjectT** obj, std::string fileName, int usize, int vsize, int windowSize) {
	const int ws = windowSize * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	const int bufferSize = BUFFERSIZE / wsTotal;
	const int totalSize = usize * vsize;

	int size[] = { usize,vsize };


	//Texture tex = Texture4DT(size[0], size[1], size[2], size[3]);
	cv::Mat texture = cv::Mat(usize, vsize, CV_64FC4);
	cv::Mat normal = cv::Mat(usize, vsize, CV_64FC3);
	cv::Mat height = cv::Mat(usize, vsize, CV_64FC1);


	double inc[] = { M_PI * 2. / size[0], M_PI / (size[1] - 1.), M_PI / (size[2] - 1.), M_PI / (size[3] - 1.) };

	typeT radius = 3.5;

	typeT denom = 1. / wsTotal;
	std::cout << "Denominador: " << denom << std::endl;

	CudaPointersT cp;
	cp.allocate(bufferSize * wsTotal, usize, vsize, 1, 1);
	{
		std::vector <cv::Vec<typeT, 4>> objColorList(NUMOBJ);
		std::vector <uint8_t> objShinyList(NUMOBJ);
		for (int i = 0; i < NUMOBJ; i++) {
			objColorList[i] = obj[i]->color;
			objShinyList[i] = obj[i]->specularShininness;
		}
		cp.uploadObjectColorProp(objColorList.data(), objShinyList.data(), NUMOBJ);
	}
	std::cout << "Allocate: " << bufferSize * wsTotal << std::endl;

	std::cout << "Compile texture" << std::endl;
	int countLoop = 0;//size[3] * (size[2] * (size[1] * 0 + 0) + 16) + 16;
	int length = bufferSize;
	if (totalSize < bufferSize)
		length = totalSize - countLoop;
	int bufferTotal;
	cv::Vec<typeT, 4>* texturePixel = (cv::Vec<typeT, 4>*)texture.data;
	vec3T* normalPixel = (vec3T *)normal.data;
	typeT* heightPixel = (typeT *)height.data;

	while (countLoop < totalSize)
	{
		auto start = std::chrono::steady_clock::now();

		bufferTotal = length * wsTotal;

		std::cout << countLoop << "/" << totalSize << std::endl;

		Texture4DT(usize, vsize, 1, 1).RayLightGeneratorCuda(countLoop, length, radius, ws, wsTotal, cp);

		cp.setHitObjectList(-1, bufferTotal);
		cp.setDistList(10e10, bufferTotal);
		for (int o = 0; o < NUMOBJ; o++)
		{
			obj[o]->CheckCollisionCuda(cp, bufferTotal, o);
		}
		LightT::setAmbientLightCUDA(cp, 1, bufferTotal);

		cp.pixelReduction(wsTotal, length);
		cp.downloadPixelColor(texturePixel, length);

		cp.normalReduction(wsTotal, length);
		cp.downloadNormalList(normalPixel, length);

		cp.distReduction(wsTotal, length);
		cp.downloadDistList(heightPixel, length);

		auto end = std::chrono::steady_clock::now();
		std::cout << " Compare Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
		//return;
		texturePixel += length;
		normalPixel += length;
		heightPixel += length;
		countLoop += length;
		if (totalSize - countLoop < bufferSize)
			length = totalSize - countLoop;
	}
	cp.free();

	std::string rootName = fileName.substr(0, rootName.length() - 4);
	std::string extension = fileName.substr(rootName.length() - 4, 4);

	cv::Mat img;
	texture.convertTo(img, CV_8UC4, 255);
	cv::transpose(img, img);
	//cv::imwrite("D:/testeTexMap.png", img);
	cv::imwrite(fileName, img);

	normal.convertTo(img, CV_8UC3, 127, 127);
	cv::transpose(img, img);
	cv::imwrite(rootName + "Normal" + extension, img);
	//cv::imwrite("D:/testeNormalMap.png", img);

	height.convertTo(img, CV_8UC1, -127 / (2 * radius), 127);
	cv::transpose(img, img);
	cv::imwrite(rootName + "Height" + extension, img);
	//cv::imwrite("D:/testeHeightMap.png", img);
}

void generateViewPortTexture(std::string textureFile, int usize, int vsize, int ssize, int tsize, float sphereRadius, vec3T viewerPos, vec3T viewerDir, typeT fov, std::string nameFile, int width, int height, int aliasing) {

	for (int i = 0; i < 20; i++)
	{

		cv::Mat img = cv::Mat(width * aliasing, height * aliasing, CV_64FC4);
		typeT halfHeight = tan(fov * 0.01745329251994329576923690768489 / 2.);

		viewerDir.normalize();
		vec3T right = vec3(-viewerDir.z, 0., viewerDir.x);
		right.normalize();
		vec3T up = vec3(-right.z * viewerDir.y, right.z * viewerDir.x - right.x * viewerDir.z, right.x * viewerDir.y);
		up.normalize();

		cv::Mat texture = cv::imread(textureFile, cv::IMREAD_UNCHANGED);
		std::cout << texture.size << std::endl;
		SphereT sphere = SphereT(vec3T(0, 0, 0), 2 * sphereRadius, cv::Vec<typeT, 3>(0, 0, 0), 0, 0);

		printf("Viewer Position: %f, %f, %f\n", viewerPos.x, viewerPos.y, viewerPos.z);
		printf("Viewer Direction: %f, %f, %f\n", viewerDir.x, viewerDir.y, viewerDir.z);
		img.forEach<cv::Vec<typeT, 4>>([&](cv::Vec<typeT, 4>& pixel, const int* pos)->void {
			typeT dV = ((1.0 - 2.0 * pos[0] / typeT(img.rows - 1)) * halfHeight);
			typeT dH = ((1.0 - 2.0 * pos[1] / typeT(img.cols - 1)) * halfHeight);

			vec3T vDaux = vec3T(1, dV, dH);
			vDaux.normalize();
			vec3T vD = vec3T(
				viewerDir.x * vDaux.x + up.x * vDaux.y + right.x * vDaux.z,
				viewerDir.y * vDaux.x + up.y * vDaux.y + right.y * vDaux.z,
				viewerDir.z * vDaux.x + up.z * vDaux.y + right.z * vDaux.z
			);

			RayLightT ray = RayLightT(viewerPos, vD);
			vec3T collision, normal;
			int objsel = -1;
			typeT dist = sphere.CheckCollision(ray, collision, normal);

			if (dist >= 0) {
				double u, v;
				u = (atan2(collision.x, collision.z) + M_PI) / (2 * M_PI) + double(i)/20;
				v = (asin(collision.y / sphereRadius) + M_PI_2) / M_PI;
				if (u < 0) u = 0;
				if (v < 0) v = 0;
				if (u > 1) u -= 1;
				if (v > 1) v = 1;
				u *= usize;
				v *= vsize;

				int ur[2], vr[2];
				ur[1] = ceil(u);
				ur[0] = ur[1] - 1;
				vr[1] = ceil(v);
				vr[0] = vr[1] - 1;

				double uw = u - ur[0] / (ur[1] - ur[0]);
				double vw = v - vr[0] / (vr[1] - vr[0]);

				cv::Vec<typeT, 4> color[4]; //uv , uV, Uv, UV
				int i = 3;
				for (int i = 0; i < 4; i++)
				{
					int new_u = ur[i / 2];
					int new_v = vr[i % 2];
					double uang = 2 * M_PI * ((double)new_u / usize) - M_PI; //[-pi, pi[
					double vang = M_PI * ((double)new_v / (vsize - 1)) - M_PI_2; //[-pi/2, pi/2]

					vec3T versorForward = vec3T(cos(uang) * cos(vang), sin(vang), sin(uang) * cos(vang));
					versorForward.normalize();
					//printf("Fw: %.2f %.2f %.2f\n", versorForward.x, versorForward.y, versorForward.z);
					vec3T versorRight = vec3T(-versorForward.z, 0., versorForward.x);
					versorRight.normalize();
					//printf("Rt: %.2f %.2f %.2f\n", versorRight.x, versorRight.y, versorRight.z);
					vec3T versorUp = vec3T(-versorRight.z * versorForward.y, versorRight.z * versorForward.x - versorRight.x * versorForward.z, versorRight.x * versorForward.y);
					versorUp.normalize();
					//printf("Up: %.2f %.2f %.2f\n\n", versorUp.x, versorUp.y, versorUp.z);

					vec3T P_point = vec3T(
						versorForward.x * vD.x + versorUp.x * vD.y + versorRight.x * vD.z,
						versorForward.y * vD.x + versorUp.y * vD.y + versorRight.y * vD.z,
						versorForward.z * vD.x + versorUp.z * vD.y + versorRight.z * vD.z
					);
					P_point.normalize();

					double sang = (atan(P_point.x / P_point.z) + M_PI_2) / M_PI;
					double tang = (asin(P_point.y) + M_PI_2) / M_PI;
					//pixel = cv::Vec<typeT, 4>(abs(versorForward.x), abs(versorForward.y), abs(versorForward.z), 1);
					//pixel = cv::Vec<typeT, 4>(versorUp.z, -versorUp.z, 0, 1);
					double s = sang * (ssize - 1);
					double t = tang * (tsize - 1);

					int sr[2], tr[2];
					sr[1] = ceil(s);
					sr[0] = sr[1] - 1;
					tr[1] = ceil(t);
					tr[0] = tr[1] - 1;

					double sw = s - sr[0] / (sr[1] - sr[0]);
					double tw = t - tr[0] / (tr[1] - tr[0]);

					cv::Vec<typeT, 4> color_01 = texture.at<cv::Vec4b>(new_v + vsize * tr[0], new_u + usize * sr[0]); //st
					color_01 /= 255;
					//std::cout << "st: " << color_01 << std::endl;
					//printf("uvst: %d %d %d %d %d %d", new_u, new_v, tr[0], sr[1], new_v + vsize * tr[0], new_u + usize * sr[1]);
					cv::Vec<typeT, 4> color_02 = texture.at<cv::Vec4b>(new_v + vsize * tr[0], new_u + usize * sr[1]); //St
					color_02 /= 255;
					//std::cout << "St: " << color_02 << std::endl;
					cv::Vec<typeT, 4> color_11 = color_01 + sw * (color_02 - color_01);  //t

					//printf("uvst: %d %d %d %d %d %d", new_u, new_v, tr[1], sr[0], new_v + vsize * tr[1], new_u + usize * sr[0]);
					color_01 = texture.at<cv::Vec4b>(new_v + vsize * tr[1], new_u + usize * sr[0]); //sT
					color_01 /= 255;
					//std::cout << "sT: " << color_01 << std::endl;
					//printf("uvst: %d %d %d %d %d %d", new_u, new_v, tr[1], sr[1], new_v + vsize * tr[1], new_u + usize * sr[1]);
					color_02 = texture.at<cv::Vec4b>(new_v + vsize * tr[1], new_u + usize * sr[1]); //ST
					color_02 /= 255;
					//std::cout << "ST: " << color_02 << std::endl;
					cv::Vec<typeT, 4> color_12 = color_01 + sw * (color_02 - color_01);  //T

					if (pos[0] == 2000 && pos[1] == 2000) {
						printf("uvst: %d %d %d %d\n", new_u, new_v, sr[0], tr[0]);
					}

					color[i] = color_11 + tw * (color_12 - color_11);
				}

				cv::Vec<typeT, 4> color_21 = color[0] + uw * (color[2] - color[0]);  // uv - Uv
				cv::Vec<typeT, 4> color_22 = color[1] + uw * (color[3] - color[1]);  // uV - UV
				cv::Vec<typeT, 4> final_color = color_21 + vw * (color_22 - color_21);
				pixel = cv::Vec<typeT, 4>(final_color[3] * final_color[0] + (1 - final_color[3]) * .5,
					final_color[3] * final_color[1] + (1 - final_color[3]) * .5,
					final_color[3] * final_color[2] + (1 - final_color[3]) * .5,
					1);
			}
			else pixel = cv::Vec4d(.5, .5, .5, 1);
		});
		//int count = 0;
		//for(int i = 0; i < img.rows; i++)
		//	for (int j = 0; j < img.cols; j++)
		//	{
		//		typeT dV = ((1.0 - 2.0 * j / typeT(img.rows - 1)) * halfHeight);
		//		typeT dH = ((1.0 - 2.0 * i / typeT(img.cols - 1)) * halfHeight);

		//		vec3T vDaux = vec3T(1, dV, dH);
		//		vDaux.normalize();
		//		vec3T vD = vec3T(
		//			viewerDir.x * vDaux.x + up.x * vDaux.y + right.x * vDaux.z,
		//			viewerDir.y * vDaux.x + up.y * vDaux.y + right.y * vDaux.z,
		//			viewerDir.z * vDaux.x + up.z * vDaux.y + right.z * vDaux.z
		//		);
		//		vD.normalize();

		//		RayLightT ray = RayLightT(viewerPos, vD);
		//		vec3T collision, normal;
		//		int objsel = -1;
		//		typeT dist = sphere.CheckCollision(ray, collision, normal);

		//		if (dist >= 0) {
		//			double u, v;
		//			vec3T dir = vD;

		//			u = (atan2(collision.x, collision.z) + M_PI) / (2 * M_PI);
		//			v = (asin(collision.y / sphereRadius) + M_PI_2) / M_PI;
		//			if (u < 0) u = 0;
		//			if (v < 0) v = 0;
		//			if (u > 1) u = 1;
		//			if (v > 1) v = 1;
		//			u *= usize;
		//			v *= vsize;

		//			int ur[2], vr[2];
		//			ur[1] = (int)ceil(u);
		//			ur[0] = ur[1] - 1;
		//			vr[1] = (int)ceil(v);
		//			vr[0] = vr[1] - 1;

		//			double uw = (u - ur[0]) / (ur[1] - ur[0]);
		//			double vw = (v - vr[0]) / (vr[1] - vr[0]);

		//			cv::Vec<typeT, 4> color[4]; //uv , uV, Uv, UV
		//			for (int i = 0; i < 4; i++)
		//			{
		//				int new_u = ur[i / 2];
		//				int new_v = vr[i % 2];
		//				double uang = 2 * M_PI * ((double)new_u / usize) - M_PI; //[-pi, pi[
		//				double vang = M_PI * ((double)new_v / (vsize - 1)) - M_PI_2; //[-pi/2, pi/2]

		//				vec3T versorForward = vec3T(cos(uang) * cos(vang), sin(vang), sin(uang) * cos(vang));
		//				versorForward.normalize();

		//				vec3T versorRight = vec3(-versorForward.z, 0., versorForward.x);
		//				versorRight.normalize();
		//				vec3T versorUp = vec3(-right.z * versorForward.y, right.z * versorForward.x - right.x * versorForward.z, right.x * versorForward.y);
		//				versorUp.normalize();

		//				vec3T P_point = vec3T(
		//					versorForward.x * dir.x + versorUp.x * dir.y + versorRight.x * dir.z,
		//					versorForward.y * dir.x + versorUp.y * dir.y + versorRight.y * dir.z,
		//					versorForward.z * dir.x + versorUp.z * dir.y + versorRight.z * dir.z
		//				);

		//				double sang = (atan(-P_point.x / P_point.z) + M_PI_2) / M_PI;
		//				double tang = (asin(P_point.y) + M_PI_2) / M_PI;

		//				if (sang < 0) sang = 0;
		//				if (tang < 0) tang = 0;
		//				if (sang > 1) sang = 1;
		//				if (tang > 1) tang = 1;
		//				double s = sang * (ssize - 1);
		//				double t = tang * (tsize - 1);
		//				//s = 16;
		//				//t = 16;

		//				int sr[2], tr[2];
		//				sr[1] = (int)ceil(s);
		//				sr[0] = sr[1] - 1;
		//				tr[1] = (int)ceil(t);
		//				tr[0] = tr[1] - 1;

		//				double sw = (s - sr[0]) / (sr[1] - sr[0]);
		//				double tw = (t - tr[0]) / (tr[1] - tr[0]);

		//				//std::cout << "sw: " << sw << " tw:" << tw << std::endl;
		//				//printf("uvst: %d %d %d %d %d %d", new_u, new_v, tr[0], sr[0], new_v + vsize * tr[0], new_u + usize * sr[0]);

		//				cv::Vec<typeT, 4> color_01 = texture.at<cv::Vec4b>(new_v + vsize * tr[0], new_u + usize * sr[0]); //st
		//				color_01 /= 255;
		//				//std::cout << "st: " << color_01 << std::endl;
		//				//printf("uvst: %d %d %d %d %d %d", new_u, new_v, tr[0], sr[1], new_v + vsize * tr[0], new_u + usize * sr[1]);
		//				cv::Vec<typeT, 4> color_02 = texture.at<cv::Vec4b>(new_v + vsize * tr[0], new_u + usize * sr[1]); //St
		//				color_02 /= 255;
		//				//std::cout << "St: " << color_02 << std::endl;
		//				cv::Vec<typeT, 4> color_11 = color_01 + sw * (color_02 - color_01);  //t

		//				//printf("uvst: %d %d %d %d %d %d", new_u, new_v, tr[1], sr[0], new_v + vsize * tr[1], new_u + usize * sr[0]);
		//				color_01 = texture.at<cv::Vec4b>(new_v + vsize * tr[1], new_u + usize * sr[0]); //sT
		//				color_01 /= 255;
		//				//std::cout << "sT: " << color_01 << std::endl;
		//				//printf("uvst: %d %d %d %d %d %d", new_u, new_v, tr[1], sr[1], new_v + vsize * tr[1], new_u + usize * sr[1]);
		//				color_02 = texture.at<cv::Vec4b>(new_v + vsize * tr[1], new_u + usize * sr[1]); //ST
		//				color_02 /= 255;
		//				//std::cout << "ST: " << color_02 << std::endl;
		//				cv::Vec<typeT, 4> color_12 = color_01 + sw * (color_02 - color_01);  //T


		//				//std::cout << "t: " << color_11 << std::endl;
		//				//std::cout << "T: " << color_12 << std::endl;

		//				color[i] = color_11 + tw * (color_12 - color_11);

		//				//std::cout << "int: " << i << " " << color[i] << std::endl;
		//			}

		//			cv::Vec<typeT, 4> color_21 = color[0] + uw * (color[2] - color[0]);  // uv - Uv
		//			cv::Vec<typeT, 4> color_22 = color[1] + uw * (color[3] - color[1]);  // uV - UV
		//			img.at < cv::Vec<typeT, 4>>(j, i) = color_21 + vw * (color_22 - color_21);
		//			count++;
		//		}
		//		else img.at<cv::Vec<typeT, 4>>(j, i) = cv::Vec4d(0, 0, 0, 0);
		//	}
		//std::cout << count << std::endl;
		cv::resize(img, img, cv::Size(width, height));
		img.convertTo(img, CV_8UC4, 255, 0);
		cv::imwrite(nameFile + std::to_string(i) + ".png", img);
	}
}

int main()
{
	ObjectT* obj[NUMOBJ];
	obj[0] = new SphereT(vec3T(0, 0, 0), 1, cv::Vec<typeT, 3>(1, 1, 1), specPower, specShiny);
	obj[1] = new SphereT(vec3T(0, 0, -1.87), 1, cv::Vec<typeT, 3>(0, 0, 1), specPower, specShiny);
	obj[2] = new CylinderT(vec3T(2.62, 0, 0), vec3T(0, 0, 0), 1, 2, cv::Vec<typeT, 3>(0.941, 0.125, 0.627), specPower, specShiny);
	obj[3] = new CubeT(vec3T(-1.2, .71, -.68), vec3T(0, 0, 0), 1, 1, 1, cv::Vec<typeT, 3>(0, 1, 0), specPower, specShiny);
	obj[4] = new CubeT(vec3T(1.47, -1.05, .93), vec3T(0, 0, 0), 1, 1, 1, cv::Vec<typeT, 3>(0, 1, 1), specPower, specShiny);
	obj[5] = new CylinderT(vec3T(-1.07, -.41, 1.81), vec3T(90, 0, 0), 1, 2, cv::Vec<typeT, 3>(1, 0, 0), specPower, specShiny);
	obj[6] = new CylinderT(vec3T(.87, .76, 2.06), vec3T(0, 0, 0), 1, 2, cv::Vec<typeT, 3>(0, 0, 0), specPower, specShiny);

	LightT* light[NUMLIGHT];
	light[0] = new DirectionalLightT(vec3T(-6, 0, 0), vec3T(1, 1, 1));

	//generateTexture(obj, light, "D:/testeGeneratorCuda2.png", U_SIZE, V_SIZE, S_SIZE, T_SIZE, 0);
	//generateTextureCuda(obj, light, "D:/testeGeneratorCuda3.png", U_SIZE, V_SIZE, S_SIZE, T_SIZE, 0);
	//generateViewPort(obj, light, vec3T(-6, 0, 0), vec3T(1, 0, 0), 90, "D:/testeView.png", 4000, 4000, 1);
	//generateViewPortCuda(obj, light, vec3T(-6, 0, 0), vec3T(1, 0, 0), 90, "D:/testeViewCuda.png", 1024, 1024, 1);
	//generateViewPort(obj, light, vec3T(12, 0, 0), vec3T(-1, 0, 0), 90, "D:/testeView2.png", 1024, 1024, 1);
	//generateMaps(obj);
	//generateMapsCuda(obj);	
	generateViewPortTexture("D:/testeGeneratorCuda.png", 512, 256, 33, 33, 3.5, vec3T(-6, 0, 0), vec3T(1, 0, 0), 90, "D:/testeViewportTexture.png", 4000, 4000, 1);

	return 0;
}

// Executar programa: Ctrl + F5 ou Menu Depurar > Iniciar Sem Depuração
// Depurar programa: F5 ou menu Depurar > Iniciar Depuração

// Dicas para Começar: 
//   1. Use a janela do Gerenciador de Soluções para adicionar/gerenciar arquivos
//   2. Use a janela do Team Explorer para conectar-se ao controle do código-fonte
//   3. Use a janela de Saída para ver mensagens de saída do build e outras mensagens
//   4. Use a janela Lista de Erros para exibir erros
//   5. Ir Para o Projeto > Adicionar Novo Item para criar novos arquivos de código, ou Projeto > Adicionar Item Existente para adicionar arquivos de código existentes ao projeto
//   6. No futuro, para abrir este projeto novamente, vá para Arquivo > Abrir > Projeto e selecione o arquivo. sln
