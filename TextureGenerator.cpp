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

struct experiment {
	int usize;
	int vsize;
	int ssize;
	int tsize;

	int windowSize;
	experiment(int usize, int vsize, int ssize, int tsize, int ws) : usize(usize), vsize(vsize), ssize(ssize), tsize(tsize), windowSize(ws) {};
	size_t totalSize() {
		return (size_t)usize * (size_t)vsize * (size_t)ssize * (size_t)tsize;
	}
};

struct viewer {
	vec3T position;
	vec3T direction;
	int width;
	int height;
	double fov;
	int aliasing;
	viewer(vec3T pos, vec3T dir, int width, int height, double fov, int alias) :position(pos), direction(dir), width(width), height(height), fov(fov), aliasing(alias) {};
};

struct pairExpIndex {
	experiment exp;
	int row;
	int col;
	pairExpIndex(experiment exp, int row, int col) : exp(exp), row(row), col(col) {}
};

struct pathway {
	vec3T initialPosition;
	vec3T finalPosition;
	vec3T direction;
	float velocity;
	pathway(vec3T initialPosition, vec3T finalPosition, vec3T direction, float velocity) : initialPosition(initialPosition), finalPosition(finalPosition), direction(direction), velocity(velocity) {}
};

struct video {
	std::vector<pathway> pathVec;
	int individualWidth;
	int individualHeight;
	double fov;
	int aliasing;
	int fps;
	std::vector<pairExpIndex> experiment;
	video(std::vector<pathway> pathVec, int individualWidth, int individualHeight, double fov, int aliasing, int fps, std::vector<pairExpIndex> experiment) :
		pathVec(pathVec), individualWidth(individualWidth), individualHeight(individualHeight), fov(fov), aliasing(aliasing), fps(fps), experiment(experiment) {}
};

cv::Mat generateViewPort(ObjectT** obj, LightT** light, vec3T viewerPos, vec3T viewerDir, typeT fov, std::string nameFile, int width, int height, int aliasing) {
	//PointLight<typeT> light = PointLight<typeT>(vec3T(-6, 4.5, 2), cv::Vec<typeT, 3>(1, 1, 1));
	
	viewerPos.x *= -1;
	viewerDir.x *= -1;
	
	cv::Mat img = cv::Mat(width * aliasing, height * aliasing, CV_64FC4);
	typeT halfHeight = tan(fov * 0.01745329251994329576923690768489 / 2.);

	viewerDir.normalize();
	vec3T right = vec3(-viewerDir.z, 0., viewerDir.x);
	right.normalize();
	vec3T up = vec3(-right.z * viewerDir.y, right.z * viewerDir.x - right.x * viewerDir.z, right.x * viewerDir.y);
	up.normalize();
	printf("Generate View Port\n");
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
		else pixel = cv::Vec4d(0.5, 0.5, 0.5, 1);
	});

	cv::resize(img, img, cv::Size(width, height));
	img.convertTo(img, CV_8UC4, 255, 0);
	std::cout << "Creating image: " << nameFile << std::endl;
	if(nameFile.compare("") != 0)
		cv::imwrite(nameFile, img);

	return img;
}

void generateViewPortCuda(ObjectT** obj, LightT** light, vec3T viewerPos, vec3T viewerDir, typeT fov, std::string nameFile, int width, int height, int aliasing) {

	cv::Mat img = cv::Mat(width * aliasing, height * aliasing, CV_64FC4);
	typeT halfHeight = tan(fov * 0.01745329251994329576923690768489 / 2.);

	viewerDir.normalize();
	vec3T right = vec3(-viewerDir.z, 0., viewerDir.x);
	right.normalize();
	vec3T up = vec3(-right.z * viewerDir.y, right.z * viewerDir.x - right.x * viewerDir.z, right.x * viewerDir.y);
	up.normalize();

	printf("Generate View Port Cuda\n");
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

	img.release();
}

void generateTexture(ObjectT** obj, LightT** light, std::string fileName, int usize, int vsize, int ssize, int tsize, int windowSize) {
	const int ws = windowSize * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	int size[] = { usize,vsize,ssize,tsize };
	Texture4DT tex = Texture4DT(size[0], size[1], size[2], size[3]);

	printf("Generate Texture\n");
	printf("Sizes: u:%d v:%d s:%d t:%d wS:%d\n", usize, vsize, ssize, tsize, windowSize);

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

static size_t countLoop = 0;//size[3] * (size[2] * (size[1] * 0 + 0) + 16) + 16;
static size_t length = 0;
static size_t bufferTotal = 0;
static size_t indexCopy = 0;
static size_t bufferSize = 0;
static size_t totalSize = 0;

void generateTextureCuda(ObjectT** obj, LightT** light, std::string fileName, size_t usize, size_t vsize, size_t ssize, size_t tsize, int windowSize) {
	const int ws = windowSize * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	std::cout << "  as as a " << fileName << std::endl;

	bufferSize = BUFFERSIZE / wsTotal;
	totalSize = usize * vsize * ssize * tsize;

	size_t size[] = { usize,vsize,ssize,tsize };
	Texture4DT tex = Texture4DT(size[0], size[1], size[2], size[3]);
	//Texture4DT tex = Texture4DT(200, 100, 1, 1, true);
	std::cout << tex.texture.total() << std::endl;

	printf("Generate Texture Cuda\n");
	printf("Sizes: u:%d v:%d s:%d t:%d wS:%d\n", usize, vsize, ssize, tsize, windowSize);

	double inc[] = { M_PI * 2. / size[0], M_PI / (size[1] - 1.), M_PI / (size[2] - 1.), M_PI / (size[3] - 1.) };

	typeT radius = 3.5;

	typeT denom = 1. / wsTotal;
	std::cout << "Denominador: " << denom << std::endl;
	std::cout << "FileName: " << fileName << std::endl;
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
	countLoop = 0;//size[3] * (size[2] * (size[1] * 0 + 0) + 16) + 16;
	length = bufferSize;
	if (totalSize - countLoop < bufferSize)
		length = totalSize - countLoop;
	cv::Vec<typeT, 4>* dataPixel = (cv::Vec<typeT, 4>*)tex.texture.data;

	int totalImage = tex.totalSize / 268435456;
	int nImage = 0;
	size_t startDown = 0;
	indexCopy = 0;

	while (countLoop < totalSize)
	{
		auto start = std::chrono::steady_clock::now();

		bufferTotal = length * wsTotal;

		auto end = std::chrono::system_clock::now();
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);

		std::cout << countLoop << "/" << totalSize << " " << (double)countLoop / totalSize * 100 << "% " << std::ctime(&end_time) << std::endl;

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

		indexCopy = startDown + length;
		if (indexCopy > tex.texture.total())
		{
			size_t dif = bufferSize - (indexCopy - tex.texture.total());
			cp.downloadPixelColor(dataPixel, dif, startDown);
			tex.compileToImage(fileName, nImage);
			nImage++;
			startDown = 0;
			cp.downloadPixelColor(dataPixel, length - dif, startDown, dif);
			startDown -= dif;
			//indexCopy = 0;
		}
		else
		{
			cp.downloadPixelColor(dataPixel, length, startDown);
		}

		//auto end = std::chrono::steady_clock::now();
		//std::cout << " Compare Elapsed time in milliseconds: "
		//	<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		//	<< " ms" << std::endl;
		//return;
		countLoop += length;
		startDown += length;
		if (totalSize - countLoop < bufferSize)
		{
			length = totalSize - countLoop;
		}
	} 
	//delete& dist;
	//delete& objSel;
	//delete& d;

	cp.free();

	{
		auto start = std::chrono::steady_clock::now();
		//tex.compileToUnity("D:\\testeGeneratorCuda.asset");
		tex.compileToImage(fileName,nImage);
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

cv::Mat generateViewPortTexture(std::string textureFile, uint usize, uint vsize, uint ssize, uint tsize, float sphereRadius, vec3T viewerPos, vec3T viewerDir, typeT fov, std::string nameFile, int width, int height, int aliasing) {

	cv::Mat img = cv::Mat::zeros(width * aliasing, height * aliasing, CV_64FC4);
	typeT halfHeight = tan(fov * 0.01745329251994329576923690768489 / 2.);

	SphereT sphere = SphereT(vec3T(0, 0, 0), 2 * sphereRadius, cv::Vec<typeT, 3>(0, 0, 0), 0, 0);

	printf("Generate View Port Texture\n");

	viewerDir.normalize();
	vec3T right = vec3(viewerDir.z, 0., -viewerDir.x);
	right.normalize();
	vec3T up = vec3(right.z * viewerDir.y, -right.z * viewerDir.x + right.x * viewerDir.z, -right.x * viewerDir.y);
	up.normalize();

	size_t totalSize = (size_t)usize * (size_t)vsize * (size_t)ssize * (size_t)tsize;
	size_t nImage = totalSize / 268435456;
	if (totalSize % 268435456 != 0)
		nImage++;

	int count = 0;
	std::cout << "Numero de texturas: " << nImage << std::endl;
	printf("Viewer Position: %f, %f, %f\n", viewerPos.x, viewerPos.y, viewerPos.z);
	printf("Viewer Direction: %f, %f, %f\n", viewerDir.x, viewerDir.y, viewerDir.z);
	for (size_t imgIndex = 0; imgIndex < nImage; imgIndex++)
	{
		std::string name;
		size_t indexfirst = textureFile.find_last_of('\\') + 1;
		size_t indexlast = textureFile.find_last_of('.');
		name = textureFile.substr(indexfirst, indexlast - indexfirst);
		name = textureFile.substr(0, indexlast);
		name += "_" + std::to_string(imgIndex) + ".png";

		std::cout << "Name " << name << " " << imgIndex << " " << std::to_string(imgIndex) << std::endl;
		Texture4DT texture = Texture4DT(name, usize, vsize, ssize, tsize);
		std::cout << "Textura " << imgIndex << " Tamanho " << texture.texture.rows << " " << texture.texture.cols << std::endl;
		std::cout << img.size << std::endl;
		//printf("right: %f, %f, %f\n", right.x, right.y, right.z);
		//printf("up: %f, %f, %f\n", up.x, up.y, up.z);

#define PARALLEL
#ifdef PARALLEL
		img.forEach<cv::Vec<typeT, 4>>([&](cv::Vec<typeT, 4>& pixel, const int* pos)->void
#else
		int pos[2];
		for (pos[0] = 0; pos[0] < img.rows; pos[0]++)
			for (pos[1] = 0; pos[1] < img.cols; pos[1]++)
#endif // PARALLEL
			{
				typeT dV = ((1.0 - 2.0 * pos[0] / typeT(img.rows - 1)) * halfHeight);
				typeT dH = -((2.0 * pos[1] / typeT(img.cols - 1) - 1) * halfHeight);

				vec3T vDaux = vec3T(dH, dV, 1);
				//printf("vDaux %f %f %f\n", vDaux.x, vDaux.y, vDaux.z);
				vDaux.normalize();
				//vec3T vD = vec3T(
				//	right.x * vDaux.x + up.x * vDaux.y + viewerDir.x * vDaux.z,
				//	right.y * vDaux.x + up.y * vDaux.y + viewerDir.y * vDaux.z,
				//	right.z * vDaux.x + up.z * vDaux.y + viewerDir.z * vDaux.z
				//);

				vec3T vD = vec3T(
					right.x * vDaux.x + up.x * vDaux.y + viewerDir.x * vDaux.z,
					right.y * vDaux.x + up.y * vDaux.y + viewerDir.y * vDaux.z,
					right.z * vDaux.x + up.z * vDaux.y + viewerDir.z * vDaux.z
				);

				RayLightT ray = RayLightT(viewerPos, vD);

				//printf("ray direction %f %f %f\n", ray.direction.x, ray.direction.y, ray.direction.z);
				vec3T collision, normal;
				typeT dist = sphere.CheckCollision(ray, collision, normal);
				vD = vD * -1;
				if (dist >= 0) {
					count++;
					double uang, vang;
					uang = atan2(collision.x, collision.z);
					vang = asin(collision.y / sphereRadius);

					//printf("\nrow: %d col: %d\n", pos[0], pos[1]);
					//printf("ray origin %f %f %f\n", ray.origin.x, ray.origin.y, ray.origin.z);
					//printf("ray direction %f %f %f\n", ray.direction.x, ray.direction.y, ray.direction.z);
					//printf("inv ray direction %f %f %f\n", vD.x, vD.y, vD.z);
					//printf("collision %f %f %f\n", collision.x, collision.y, collision.z);
					//printf("normal %f %f %f\n", normal.x, normal.y, normal.z);
					//printf("uang: %f vang: %f\n", uang * 180 / M_PI, vang * 180 / M_PI);

					uang = usize * (uang + M_PI) / (2 * M_PI);
					vang = (vsize - 1) * (vang + M_PI_2) / M_PI;

					int u[2], v[2];
					u[1] = ceil(uang);
					u[0] = u[1] - 1;
					if (u[1] == usize) u[1] = 0;
					v[1] = ceil(vang);
					v[0] = v[1] - 1;

					double uw = uang - u[0], vw = vang - v[0];

					//printf("u: %d %d %f v: %d %d %f\n\n", u[0], u[1], uang, v[0], v[1], vang);

					cv::Vec<typeT, 4> color[4];
					for (int i = 0; i < 4; i++) { //uv, Uv, uV, UV
						int u_win = u[i % 2];
						int v_win = v[i / 2];
						//printf("i iter %d u: %d v: %d\n", i, u_win, v_win);
						double uang = 2 * M_PI * u_win / usize - M_PI;
						double vang = M_PI * v_win / (vsize - 1) - M_PI_2;
						//printf("i iter %d u: %f v: %f\n", i, uang * 180 / M_PI, vang * 180 / M_PI);

						vec3T versorForward = vec3T(sin(uang) * cos(vang), sin(vang), cos(uang) * cos(vang));
						versorForward.normalize();
						//printf("forward %f %f %f\n", versorForward.x, versorForward.y, versorForward.z);
						vec3T versorRight = vec3T(versorForward.z, 0., -versorForward.x);
						versorRight.normalize();
						vec3T versorUp = vec3T(versorRight.z * versorForward.y, -versorRight.z * versorForward.x + versorRight.x * versorForward.z, -versorRight.x * versorForward.y);
						versorUp.normalize();
						//printf("right %f %f %f\n", versorRight.x, versorRight.y, versorRight.z);
						//printf("up %f %f %f\n", versorUp.x, versorUp.y, versorUp.z);

						vec3T vD_P = vec3T(
							versorRight.x * vD.x + versorRight.y * vD.y + versorRight.z * vD.z,
							versorUp.x * vD.x + versorUp.y * vD.y + versorUp.z * vD.z,
							versorForward.x * vD.x + versorForward.y * vD.y + versorForward.z * vD.z
						);
						vD_P.normalize();
						//printf("new Dir %f %f %f\n", vD_P.x, vD_P.y, vD_P.z);

						double sang = atan(vD_P.x / vD_P.z);
						double tang = asin(vD_P.y);
						//printf("sang: %f tang: %f\n", sang * 180 / M_PI, tang * 180 / M_PI);

						sang = (ssize - 1) * (sang + M_PI_2) / M_PI;
						tang = (tsize - 1) * (tang + M_PI_2) / M_PI;

						int s[2], t[2];
						s[1] = ceil(sang);
						s[0] = s[1] - 1;
						t[1] = ceil(tang);
						t[0] = t[1] - 1;
						//printf("s: %d %d %f t: %d %d %f\n\n", s[0], s[1], sang, t[0], t[1], tang);

						double sw = sang - s[0], tw = tang - t[0];

						size_t coord = texture.getCoord(u_win, v_win, s[0], t[0]) - imgIndex * (size_t)268435456;
						cv::Vec<typeT, 4> color_01, color_02;
						if (coord >= 0 && coord < 268435456)
						{
							color_01 = texture.texture.at<cv::Vec4b>(coord);
							color_01 /= 255;
						}
						else
							color_01 = cv::Vec<typeT, 4>(0, 0, 0, 0);

						coord = texture.getCoord(u_win, v_win, s[1], t[0]) - imgIndex * (size_t)268435456;
						if (coord >= 0 && coord < 268435456)
						{
							color_02 = texture.texture.at<cv::Vec4b>(coord);
							color_02 /= 255;
						}
						else
							color_02 = cv::Vec<typeT, 4>(0, 0, 0, 0);
						cv::Vec<typeT, 4> color_11 = color_01 + sw * (color_02 - color_01);  //t

						coord = texture.getCoord(u_win, v_win, s[0], t[1]) - imgIndex * (size_t)268435456;
						if (coord >= 0 && coord < 268435456)
						{
							color_01 = texture.texture.at<cv::Vec4b>(coord);
							color_01 /= 255;
						}
						else
							color_01 = cv::Vec<typeT, 4>(0, 0, 0, 0);

						coord = texture.getCoord(u_win, v_win, s[1], t[1]) - imgIndex * (size_t)268435456;
						if (coord >= 0 && coord < 268435456)
						{
							color_02 = texture.texture.at<cv::Vec4b>(coord);
							color_02 /= 255;
						}
						else
							color_02 = cv::Vec<typeT, 4>(0, 0, 0, 0);
						cv::Vec<typeT, 4> color_12 = color_01 + sw * (color_02 - color_01);  //T

						color[i] = color_11 + tw * (color_12 - color_11);
					}

					cv::Vec<typeT, 4> color_21 = color[0] + uw * (color[1] - color[0]);  // uv - Uv
					cv::Vec<typeT, 4> color_22 = color[2] + uw * (color[3] - color[2]);  // uV - UV
					cv::Vec<typeT, 4> final_color = color_21 + vw * (color_22 - color_21);
					/*pixel = cv::Vec<typeT, 4>(final_color[3] * final_color[0] + (1 - final_color[3]) * .5,
						final_color[3] * final_color[1] + (1 - final_color[3]) * .5,
						final_color[3] * final_color[2] + (1 - final_color[3]) * .5,
						1);*/


#ifdef PARALLEL
					pixel += final_color;
				}
			});
#else
					img.at<cv::Vec4d>(pos) = final_color;
				}
			}
#endif
				//else pixel = cv::Vec4d(0.5,0.5,0.5,1);
			//}
		texture.texture.release();
	}

	img.forEach<cv::Vec<typeT, 4>>([&](cv::Vec<typeT, 4>& pixel, const int* pos)->void
		{
			cv::Vec<double, 4> color = pixel;
			pixel = cv::Vec<typeT, 4>(color[3] * color[0] + (1 - color[3]) * .5,
				color[3] * color[1] + (1 - color[3]) * .5,
				color[3] * color[2] + (1 - color[3]) * .5,
				1);
		});

	std::cout << "Count Collision: " << count << std::endl;
	cv::resize(img, img, cv::Size(width, height), 0, 0, cv::INTER_AREA);
	img.convertTo(img, CV_8UC4, 255, 0);
	if(nameFile.compare("") != 0)
		cv::imwrite(nameFile, img);
	std::cout << std::endl << std::endl;
	return img;
	//img.release();
}

void generateVideo(video vid, std::string nameFile, std::string filePaste, ObjectT** obj, LightT** light)
{
	const int borderSize = 1;

	int maxCol = 0, maxRow = 0;
	for (pairExpIndex exp : vid.experiment)
	{
		if (maxCol < exp.col)
			maxCol = exp.col;
		if (maxRow < exp.row)
			maxRow = exp.row;
	}
	cv::Size videoSize = cv::Size((maxCol + 1) * vid.individualWidth + (2 + maxCol) * borderSize, (maxRow + 1) * vid.individualHeight + (2 + maxRow) * borderSize);
	cv::VideoWriter vw = cv::VideoWriter(nameFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), vid.fps, videoSize);

	cv::Mat image = cv::Mat::zeros(videoSize, CV_8UC3);
	for (pathway& path : vid.pathVec)
	{
		vec3T pos = path.initialPosition;
		vec3T endPos = path.finalPosition;
		vec3T disp_frame = endPos - pos;
		long number_frame = sqrt(disp_frame.dot(disp_frame)) * (vid.fps / path.velocity);
		disp_frame.normalize();
		disp_frame = disp_frame * (path.velocity / vid.fps);
		std::cout << "disp_frame: " << disp_frame.x << " " << disp_frame.y << " " << disp_frame.z << std::endl;

		cv::Mat frame_img;
		for (long frame = 0; frame < number_frame; frame++)
		{
			std::cout << "Position: " << pos.x << " " << pos.y << " " << pos.z << std::endl;
			for (pairExpIndex experiment : vid.experiment)
			{
				std::cout << experiment.row << " " << experiment.col << std::endl;
				if (experiment.exp.totalSize() == 0)
				{
					frame_img = generateViewPort(obj, light, pos, path.direction, vid.fov, "", vid.individualWidth, vid.individualHeight, vid.aliasing);
					cv::putText(frame_img, "Direct Render", cv::Point(5, vid.individualHeight - 32), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
				}
				else
				{
					std::string textureName = "texture_u" + std::to_string(experiment.exp.usize) + "_v" + std::to_string(experiment.exp.vsize) +
						"_s" + std::to_string(experiment.exp.ssize) + "_t" + std::to_string(experiment.exp.tsize) + "_ws" + std::to_string(experiment.exp.windowSize) + ".png";
					frame_img = generateViewPortTexture(filePaste + textureName, experiment.exp.usize, experiment.exp.vsize, experiment.exp.ssize, experiment.exp.tsize, 3.5, pos, path.direction, vid.fov, "", vid.individualWidth, vid.individualHeight, vid.aliasing);
					std::string sizeString = std::to_string(experiment.exp.usize) + "x" + std::to_string(experiment.exp.vsize) +
						"x" + std::to_string(experiment.exp.ssize) + "x" + std::to_string(experiment.exp.tsize);
					cv::putText(frame_img, sizeString, cv::Point(5, vid.individualHeight - 32), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
				}
				cv::Rect rect = cv::Rect(vid.individualWidth * experiment.col + (1 + experiment.col) * borderSize, vid.individualHeight * experiment.row + (1 + experiment.row) * borderSize, vid.individualWidth, vid.individualHeight);
				std::cout << experiment.row << " " << experiment.col << " " << rect << std::endl;
				cv::cvtColor(frame_img, frame_img, cv::COLOR_BGRA2BGR);
				frame_img.copyTo(image(rect));
			}
			///cv::imwrite("D:\\TextureGenerator\\NewImage\\teste.jpg", image);
			vw.write(image);
			pos = pos + disp_frame;
			std::cout << "Position2: " << pos.x << " " << pos.y << " " << pos.z << std::endl;
		}
	}
}

int main(int argc, char *argv[])
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

	// < 2^31
	std::vector<experiment> expList = {
		//experiment(512,256,64,64,0),
		//experiment(512,256,64,32,0),
		//experiment(128,64,16,16,3),
		// experiment(1024,512,64,64,3),
		experiment(1024,512,128,128,3),
		//experiment(512,256,32,32,0),
		//experiment(512,256,64,64,3),
		//experiment(512,256,32,32,3),
		//experiment(512,256,32,32,0),
		//experiment(512,256,32,32,3),
		//experiment(64,32,256,256,0),
		//experiment(64,32,256,256,3),
		//experiment(64,32,256,256,0),
		//experiment(64,32,256,256,3),
		//experiment(512,256,64,64,0),
		//experiment(512,256,64,64,3),
		//experiment(1024,512,64,32,0),
		//experiment(1024,512,64,32,3), 
		experiment(1024,512,128,128,3),
		experiment(512,256,128,128,3),
		experiment(512,256,128,128,0),
		//experiment(1024,512,64,64,0),
	};

	std::vector<viewer> viewList = {
		viewer(vec3T(-3, 0, 10),vec3T(0, 0, -1),512,512,60,1),
		viewer(vec3T(0, 0, 10),vec3T(0, 0, -1),512,512,60,1),
		viewer(vec3T(3, 0, 10),vec3T(0, 0, -1),512,512,60,1),
		viewer(vec3T(-3, 0, -10),vec3T(0, 0, 1),512,512,60,1),
		viewer(vec3T(0, 0, -10),vec3T(0, 0, 1),512,512,60,1),
		viewer(vec3T(3, 0, -10),vec3T(0, 0, 1),512,512,60,1),

		viewer(vec3T(10, 0, -3),vec3T(-1, 0, 0),512,512,60,1),
		viewer(vec3T(10, 0, 0),vec3T(-1, 0, 0),512,512,60,1),
		viewer(vec3T(10, 0, 3),vec3T(-1, 0, 0),512,512,60,1),
		viewer(vec3T(-10, 0, -3),vec3T(1, 0, 0),512,512,60,1),
		viewer(vec3T(-10, 0, 0),vec3T(1, 0, 0),512,512,60,1),
		viewer(vec3T(-10, 0, 3),vec3T(1, 0, 0),512,512,60,1),

		viewer(vec3T(0, 10, 0),vec3T(0,-1, 0),512,512,60,1),
		viewer(vec3T(0, -10, 0),vec3T(0, 1, 0),512,512,60,1),

		//viewer(vec3T(-3, 0, 10),vec3T(0, 0, -1),512,512,60,50),
		//viewer(vec3T(0, 0, 10),vec3T(0, 0, -1),512,512,60,50),
		//viewer(vec3T(3, 0, 10),vec3T(0, 0, -1),512,512,60,50),
		//viewer(vec3T(-3, 0, -10),vec3T(0, 0, 1),512,512,60,5),
		//viewer(vec3T(0, 0, -10),vec3T(0, 0, 1),512,512,60,5),
		//viewer(vec3T(3, 0, -10),vec3T(0, 0, 1),512,512,60,5),

		//viewer(vec3T(10, 0, -3),vec3T(-1, 0, 0),512,512,60,5),
		viewer(vec3T(10, 0, 0),vec3T(-1, 0, 0),512,512,60,5),
		//viewer(vec3T(10, 0, 3),vec3T(-1, 0, 0),512,512,60,5),
		//viewer(vec3T(-10, 0, -3),vec3T(1, 0, 0),512,512,60,5),
		//viewer(vec3T(-10, 0, 0),vec3T(1, 0, 0),512,512,60,5),
		//viewer(vec3T(-10, 0, 3),vec3T(1, 0, 0),512,512,60,5),
	};

	video vidInfo = video(
		std::vector<pathway>({
			pathway(vec3T(-3, 0, 10), vec3T(3, 0, 10), vec3T(0, 0, -1), 1.f)
			}),
		512, 512, 60, 5, 24,
		std::vector<pairExpIndex>({
		pairExpIndex(experiment(0,0,0,0,0),0,0),
		pairExpIndex(experiment(512,256,64,64,3),0,1)
		}));

	//std::string filePaste = "D:/teste/generator/";
	std::string filePaste = argv[1];

	for (int i = 0; i < expList.size(); i++) {
	//for (int i = 0; i < 1; i++) {
		std::cout << "Experiment " << i << std::endl;
		experiment exp = expList[i];
		std::string textureName = "texture_u" + std::to_string(exp.usize) + "_v" + std::to_string(exp.vsize) +
			"_s" + std::to_string(exp.ssize) + "_t" + std::to_string(exp.tsize) + "_ws" + std::to_string(exp.windowSize) + ".png";

		//Gera Textura
		//generateTextureCuda(obj, light, filePaste+textureName, exp.usize, exp.vsize, exp.ssize, exp.tsize, exp.windowSize);

		//Gera ViewPort com base na textura
		for (int j = 0; j < viewList.size(); j++) {
			std::cout << "Viewer Texture " << j << std::endl;
			viewer view = viewList[j];
			std::string fileName = filePaste + textureName +
				"__pos_" + (view.position.x < 0 ? "m" + std::to_string(abs(view.position.x)) : std::to_string(view.position.x)) +
				"_" + (view.position.y < 0 ? "m" + std::to_string(abs(view.position.y)) : std::to_string(view.position.y)) +
				"_" + (view.position.z < 0 ? "m" + std::to_string(abs(view.position.z)) : std::to_string(view.position.z)) +
				"__dir_" + (view.direction.x < 0 ? "m" + std::to_string(abs(view.direction.x)) : std::to_string(view.direction.x)) +
				"_" + (view.direction.y < 0 ? "m" + std::to_string(abs(view.direction.y)) : std::to_string(view.direction.y)) +
				"_" + (view.direction.z < 0 ? "m" + std::to_string(abs(view.direction.z)) : std::to_string(view.direction.z)) +
				"__al_" + std::to_string(view.aliasing);
			//generateViewPort(obj, light, view.position, view.direction, view.fov, fileName + "rend.png", view.width, view.height, view.aliasing);
			//generateViewPortTexture(filePaste+textureName, exp.usize, exp.vsize, exp.ssize, exp.tsize, 3.5, view.position, view.direction, view.fov, fileName + "text.png", view.width, view.height, view.aliasing);
		}
	}

	//Gera ViewPort com base nos object render
	for (int j = 0; j < viewList.size(); j++) {
		std::cout << "Viewer Render " << j << std::endl;
		viewer view = viewList[j];
		std::string fileName = filePaste + "rend" +
			"__pos_" + (view.position.x < 0 ? "m" + std::to_string(abs(view.position.x)) : std::to_string(view.position.x)) +
			"_" + (view.position.y < 0 ? "m" + std::to_string(abs(view.position.y)) : std::to_string(view.position.y)) +
			"_" + (view.position.z < 0 ? "m" + std::to_string(abs(view.position.z)) : std::to_string(view.position.z)) +
			"__dir_" + (view.direction.x < 0 ? "m" + std::to_string(abs(view.direction.x)) : std::to_string(view.direction.x)) +
			"_" + (view.direction.y < 0 ? "m" + std::to_string(abs(view.direction.y)) : std::to_string(view.direction.y)) +
			"_" + (view.direction.z < 0 ? "m" + std::to_string(abs(view.direction.z)) : std::to_string(view.direction.z)) +
			"__al_" + std::to_string(view.aliasing);
		//generateViewPort(obj, light, view.position, view.direction, view.fov, fileName + ".png", view.width, view.height, view.aliasing);
	}

	generateVideo(vidInfo, filePaste + "demonstration_video.avi", filePaste, obj, light);

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
