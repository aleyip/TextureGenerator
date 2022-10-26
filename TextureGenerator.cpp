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

#include "Object.h"
#include "Sphere.h"
#include "Cylinder.h"
#include "Cube.h"
#include "RayLight.h"
#include "Texture4D.h"

#include "CudaPointers.h"

#define NUMOBJ 7
#define WINDOW_SIZE 3
#define BUFFERSIZE 100000000

#define TEXU 512
#define TEXV 256
#define TEXS 33
#define TEXT 33

using typeT = double;
using ObjectT = Object<typeT>;
using SphereT = Sphere<typeT>;
using CylinderT = Cylinder<typeT>;
using CubeT = Cube<typeT>;
using RayLightT = RayLight<typeT>;
using Texture4DT = Texture4D<typeT>;
using vec3T = vec3<typeT>;
using CudaPointersT = CudaPointers<typeT>;

void generateViewPort(ObjectT** obj) {
	cv::Mat img = cv::Mat(2048, 2048, CV_64FC4);
	typeT fov = 90;
	typeT halfHeight = tan(fov * 0.01745329251994329576923690768489 / 2.);
	vec3T viewPos = vec3T(-12, 0, 0);
	vec3T viewDir = vec3T(1, 0, 0);

	viewDir.normalize();
	vec3T right = vec3(-viewDir.z, 0., viewDir.x);
	right.normalize();
	vec3T up = vec3(-right.z * viewDir.y, right.z * viewDir.x - right.x * viewDir.z, right.x * viewDir.y);
	up.normalize();

	printf("%f, %f, %f\n", viewDir.x, viewDir.y, viewDir.z);
	printf("%f, %f, %f\n", viewPos.x, viewPos.y, viewPos.z);
	img.forEach<cv::Vec<typeT, 4>>([&obj, &viewPos, &viewDir, img, halfHeight, &right, &up](cv::Vec<typeT, 4>& pixel, const int* pos)->void {
		typeT dV = ((1.0 - 2.0 * pos[0] / typeT(img.rows - 1)) * halfHeight);
		typeT dH = ((1.0 - 2.0 * pos[1] / typeT(img.cols - 1)) * halfHeight);

		vec3T vDaux = vec3T(1, dV, dH);
		vDaux.normalize();
		vec3T vD = vec3T(
			viewDir.x * vDaux.x + up.x * vDaux.y + right.x * vDaux.z,
			viewDir.y * vDaux.x + up.y * vDaux.y + right.y * vDaux.z,
			viewDir.z * vDaux.x + up.z * vDaux.y + right.z * vDaux.z
		);

		RayLightT ray = RayLightT(viewPos, vD);
		vec3T normal, collision;
		ObjectT* colObj[1];
		typeT dist = -1;
		for (int i = 0; i < NUMOBJ; i++)
		{
			typeT d = obj[i]->CheckCollision(ray, collision, normal);
			//typeT d = obj[i]->CheckCollision(ray);
			if ((d < dist && d >= 0) || dist == -1) {
				dist = d;
				colObj[0] = obj[i];
			}
		}

		if (dist >= 0)pixel = colObj[0]->color;
		else pixel = cv::Vec4d(0, 0, 0, 0);
	});

	cv::resize(img, img, cv::Size(720, 720));
	img.convertTo(img, CV_8UC4, 255, 0);
	cv::imshow("teste", img);
	cv::imwrite("D:/testeGen.png", img);
	cv::waitKey(0);
}

void generateViewPortCuda(ObjectT** obj) {
	cv::Mat img = cv::Mat(2048, 2048, CV_64FC4);
	typeT fov = 90;
	typeT halfHeight = tan(fov * 0.01745329251994329576923690768489 / 2.);
	vec3T viewPos = vec3T(-12, 0, 0);
	vec3T viewDir = vec3T(1, 0, 0);

	viewDir.normalize();
	vec3T right = vec3(-viewDir.z, 0., viewDir.x);
	right.normalize();
	vec3T up = vec3(-right.z * viewDir.y, right.z * viewDir.x - right.x * viewDir.z, right.x * viewDir.y);
	up.normalize();

	printf("%f, %f, %f\n", viewDir.x, viewDir.y, viewDir.z);
	printf("%f, %f, %f\n", viewPos.x, viewPos.y, viewPos.z);

	CudaPointersT cp;
	cp.allocate(img.total(), 0, 0, 0, 0);

	std::vector<RayLightT> rayList = std::vector<RayLightT>(img.total());
	for (int r = 0; r < img.rows; r++)
		for (int c = 0; c < img.cols; c++)
		{
			typeT dV = ((1.0 - 2.0 * r / typeT(img.rows - 1)) * halfHeight);
			typeT dH = ((1.0 - 2.0 * c / typeT(img.cols - 1)) * halfHeight);

			vec3T vDaux = vec3T(1, dV, dH);
			vDaux.normalize();
			vec3T vD = vec3T(
				viewDir.x * vDaux.x + up.x * vDaux.y + right.x * vDaux.z,
				viewDir.y * vDaux.x + up.y * vDaux.y + right.y * vDaux.z,
				viewDir.z * vDaux.x + up.z * vDaux.y + right.z * vDaux.z
			);

			//if ((r == 1024 || r == 0) && c == 1024) {
			//	printf("a %d %d %f, %f, %f\n", r, c, vDaux.x, vDaux.y, vDaux.z);
			//	printf("b %d %d %f, %f, %f\n", r, c, vD.x, vD.y, vD.z);
			//}

			rayList[img.rows * c + r] = RayLightT(viewPos, vD);
		}

	std::vector<typeT>distList = std::vector<typeT>(img.total(), -1);
	std::vector<int>selObjList = std::vector<int>(img.total(), -1);

	cp.uploadRayList(rayList);
	for (int j = 0; j < NUMOBJ; j++)
	{
		std::vector<typeT> dList = std::vector<typeT>(rayList.size());
		obj[j]->CheckCollisionCuda(dList, cp);
		for(int i = 0; i < img.total(); i++)
			if ((dList[i] < distList[i] && dList[i] >= 0) || distList[i] == -1) {
				distList[i] = dList[i];
				selObjList[i] = j;
			}
	}

	cp.free();

	for (int r = 0; r < img.rows; r++)
		for (int c = 0; c < img.cols; c++)
		{
			if (distList[img.rows * c + r] >= 0)img.at<cv::Vec4d>(r, c) = obj[selObjList[img.rows * c + r]]->color;
			else img.at<cv::Vec4d>(r, c) = cv::Vec4d(0, 0, 0, 0);
		}

	cv::resize(img, img, cv::Size(720, 720));
	img.convertTo(img, CV_8UC4, 255, 0);
	cv::imshow("teste2", img);
	cv::imwrite("D:/testeGenCuda.png", img);
	cv::waitKey(0);
}

void generateTexture(ObjectT** obj) {
	const int ws = WINDOW_SIZE * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	int size[] = { TEXU,TEXV,TEXS,TEXT };
	Texture4DT tex = Texture4DT(size[0], size[1], size[2], size[3]);

	double inc[] = { M_PI * 2. / size[0], M_PI / (size[1] - 1.), M_PI / (size[2] - 1.), M_PI / (size[3] - 1.) };
	double halfinc[4];
	if (WINDOW_SIZE == 0) {
		halfinc[0] = 0;
		halfinc[1] = 0;
		halfinc[2] = 0;
		halfinc[3] = 0;
	}
	else {
		halfinc[0] = inc[0] / (2. * WINDOW_SIZE);
		halfinc[1] = inc[1] / (2. * WINDOW_SIZE);
		halfinc[2] = inc[2] / (2. * WINDOW_SIZE);
		halfinc[3] = inc[3] / (2. * WINDOW_SIZE);
	}

	typeT radius = 3.5;

	typeT denom = 1. / wsTotal;
	std::cout << denom << std::endl;

	std::cout << "Compile Texture" << std::endl;
	auto start = std::chrono::steady_clock::now();
	for (int t = 0; t < size[3]; t++) {
		auto start = std::chrono::steady_clock::now();
		std::cout << "Step t" << t << std::endl;
		for (int s = 0; s < size[2]; s++) {
			std::cout << "Step t" << t << " s " << s << std::endl;
			auto start = std::chrono::steady_clock::now();
#pragma omp parallel for collapse(2)
			for (int v = 0; v < size[1]; v++) {
				for (int u = 0; u < size[0]; u++)
				{
					double angleBase[] = { -(inc[0] * u - M_PI), inc[1] * v - M_PI_2, inc[2] * s - M_PI_2, inc[3] * t - M_PI_2 };

					cv::Vec<typeT, 4> sumPixel = cv::Vec<typeT, 4>(0, 0, 0, 0);

					int ind = 0;
					//#pragma omp parallel for collapse(4)
					for (int n = -WINDOW_SIZE; n <= WINDOW_SIZE; n++)
						for (int m = -WINDOW_SIZE; m <= WINDOW_SIZE; m++)
							for (int l = -WINDOW_SIZE; l <= WINDOW_SIZE; l++)
								for (int k = -WINDOW_SIZE; k <= WINDOW_SIZE; k++)
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

									cv::Vec4d color;
									double dist = -1;
									for (int i = 0; i < NUMOBJ; i++)
									{
										double d = obj[i]->CheckCollision(ray, collision, normal);
										if ((d < dist && d >= 0) || dist == -1) {
											dist = d;
											color = obj[i]->color;
										}
										//printf("dist: %f %f ", d, dist);
									}
									if (dist >= 0)sumPixel += color;
									//std::cout << std::endl;
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

	cv::Mat img = cv::Mat(256, 512, CV_64FC4);
	for (int u = 0; u < 512; u++)
		for (int v = 0; v < 256; v++)
			img.at<cv::Vec4d>(v, u) = tex(u, v, 16, 16);
	img.convertTo(img, CV_8UC4, 255);
	cv::imshow("Teste", img);
	cv::imwrite("D:/testeTexGen.png", img);
	cv::waitKey(0);

	{
		auto start = std::chrono::steady_clock::now();
		tex.compileToUnity("D:\\testeGenerator.asset");
		auto end = std::chrono::steady_clock::now();
		std::cout << "Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
	}
}

void generateTextureCuda(ObjectT** obj) {
	const int ws = WINDOW_SIZE * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	const int bufferSize = BUFFERSIZE / wsTotal;
	const int totalSize = TEXU * TEXV * TEXS * TEXT;

	int size[] = { TEXU,TEXV,TEXS,TEXT };
	Texture4DT tex = Texture4DT(size[0], size[1], size[2], size[3]);

	double inc[] = { M_PI * 2. / size[0], M_PI / (size[1] - 1.), M_PI / (size[2] - 1.), M_PI / (size[3] - 1.) };

	typeT radius = 3.5;

	typeT denom = 1. / wsTotal;
	std::cout << "Denominador: " << denom << std::endl;

	//std::vector<RayLightT> rayList = std::vector< RayLightT>(totalSize);
	std::vector<typeT> dist = std::vector< typeT>(bufferSize * wsTotal);
	std::vector<int> objSel = std::vector<int>(bufferSize * wsTotal);
	std::vector<typeT> d = std::vector<typeT>(bufferSize * wsTotal);

	CudaPointersT cp;
	std::cout << "Allocate: " << bufferSize * wsTotal << std::endl;
	cp.allocate(bufferSize * wsTotal, TEXU, TEXV, TEXS, TEXT);

	std::cout << "Compile Texture" << std::endl;
	int countLoop = 0;//size[3] * (size[2] * (size[1] * 0 + 0) + 16) + 16;
	int length = bufferSize;
	while (countLoop < totalSize)
	{
		auto start = std::chrono::steady_clock::now();

		std::cout << countLoop << "/" << totalSize;

		tex.RayLightGeneratorCuda(countLoop, length, radius, ws, wsTotal, cp);
		//std::vector<RayLightT> rayList = std::vector<RayLightT>(bufferSize * wsTotal);
		//cp.downloadRayList(rayList);
		//for (int i = 0; i < rayList.size(); i++) {
		//	int index = countLoop + i / wsTotal;
		//	int u = (index) / (size[3] * size[2] * size[1]);
		//	int v = ((index / size[3]) / size[2]) % size[1];
		//	int s = (index / size[3]) % size[2];
		//	int t = index % size[3];
		//	if (u == 256 && v == 128 && s == 0 && t == 0) {
		//		printf("u: %d v : %d s: %d t: %d ", index / (size[3] * size[2] * size[1]), ((index / size[3]) / size[2]) % size[1], (index / size[3]) % size[2], index % size[3]);
		//		printf("%d pos: %f %f %f dir: %f %f %f\n", i, rayList[i].origin.x, rayList[i].origin.y, rayList[i].origin.z, rayList[i].direction.x, rayList[i].direction.y, rayList[i].direction.z);
		//	}
		//}
		//auto end = std::chrono::steady_clock::now();
		//std::cout << "Ray Light Elapsed time in milliseconds: "
		//	<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		//	<< " ms" << std::endl;

		//auto start1 = std::chrono::steady_clock::now();
		for (int& i : objSel)
			i = -1;
		for (typeT& i : dist)
			i = -1;
		for (int o = 0; o < NUMOBJ; o++)
		{
			obj[o]->CheckCollisionCuda(d, cp);
#pragma omp parallel for
			for (int i = 0; i < d.size(); i++)
				if ((d[i] < dist[i] && d[i] >= 0) || dist[i] == -1) {
					dist[i] = d[i];
					objSel[i] = o;
				}
		}

#pragma omp parallel for
		for (int i = 0; i < length; i++)
		{
			int u = (countLoop + i) / (size[3] * size[2] * size[1]);
			int v = (((countLoop + i) / size[3]) / size[2]) % size[1];
			int s = ((countLoop + i) / size[3]) % size[2];
			int t = (countLoop + i) % size[3];

			cv::Vec4d sumPixel = cv::Vec4d(0, 0, 0, 0);
			for (int j = 0; j < wsTotal; j++)
			{
				//if ((u == 41 && v == 156 && s == 0 && t == 0) || (u == 300 && v == 101 && s == 0 && t == 0)) {
				//	printf("u: %d v: %d s: %d t: %d ", u, v, s, t);
				//	std::cout << i << " " << j << " " << wsTotal * i + j << " " << dist[wsTotal * i + j] << " " << objSel[wsTotal * i + j] << std::endl;
				//}
				if (dist[wsTotal * i + j] >= 0)
				{
					sumPixel += obj[objSel[wsTotal * i + j]]->color;/*
					if (u == 256 && v == 128 && s == 0 && t == 0) {
						std::cout << " Color: " << color << sumPixel << std::endl;
					}*/
				}
				//else if (u == 256 && v == 128 && s == 0 && t == 0) std::cout << " No hit" << std::endl;
			}
			tex.texture.at<cv::Vec<typeT,4>>(countLoop + i) = sumPixel * denom;
			tex(u, v, s, t) = sumPixel * denom;
			//if ((u == 41 && v == 156 && s == 0 && t == 0) || (u == 300 && v == 101 && s == 0 && t == 0))
			//	std::cout << " Final Color: " << sumPixel * denom << tex(u, v, s, t) << std::endl;

		}

		auto end = std::chrono::steady_clock::now();
		std::cout << " Compare Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
		//return;

		countLoop += length;
		if (totalSize - countLoop < bufferSize)
			length = totalSize - countLoop;
	} 
	//delete& dist;
	//delete& objSel;
	//delete& d;

	cp.free();

	cv::Mat img = cv::Mat(TEXV, TEXU, CV_64FC4);
	for (int u = 0; u < TEXU; u++)
		for (int v = 0; v < TEXV; v++)
			//img.at<cv::Vec4d>(v, u) = tex(u, v, 0, 0);
			img.at<cv::Vec4d>(v, u) = tex(u, v, TEXS / 2, TEXT / 2);
	img.convertTo(img, CV_8UC4, 255);
	cv::imwrite("D:/testeGen.png", img);
	cv::imshow("Teste", img);
	cv::imwrite("D:/testeTexGenCuda.png", img);
	cv::waitKey(0);

	{
		auto start = std::chrono::steady_clock::now();
		tex.compileToUnity("D:\\testeGeneratorCuda.asset");
		auto end = std::chrono::steady_clock::now();
		std::cout << "Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
	}
}

int main()
{
	const int ws = WINDOW_SIZE * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	ObjectT* obj[NUMOBJ];
	obj[0] = new SphereT(vec3T(0, 0, 0), 1, cv::Vec4d(1, 1, 1, 1));
	obj[1] = new SphereT(vec3T(0, 0, -1.87), 1, cv::Vec4d(0, 0, 1, 1));
	obj[2] = new CylinderT(vec3T(2.62, 0, 0), vec3T(0, 0, 0), 1, 2, cv::Vec4d(0.941, 0.125, 0.627, 1));
	obj[3] = new CubeT(vec3T(-1.2, .71, -.68), vec3T(0, 0, 0), 1, 1, 1, cv::Vec4d(0, 1, 0, 1));
	obj[4] = new CubeT(vec3T(1.47, -1.05, .93), vec3T(0, 0, 0), 1, 1, 1, cv::Vec4d(0, 1, 1, 1));
	obj[5] = new CylinderT(vec3T(-1.07, -.41, 1.81), vec3T(0, 0, 0), 1, 2, cv::Vec4d(1, 0, 0, 1));
	obj[6] = new CylinderT(vec3T(.87, .76, 2.06), vec3T(0, 0, 0), 1, 2, cv::Vec4d(0, 0, 0, 1));

	//generateTexture(obj);
	//generateTextureCuda(obj);
	generateViewPort(obj);
	generateViewPortCuda(obj);

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
