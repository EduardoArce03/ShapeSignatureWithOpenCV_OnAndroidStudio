#ifndef VISIO_P3_HU_UTILS_H
#define VISIO_P3_HU_UTILS_H

#include <vector>
#include <string>
#include <android/asset_manager.h>
#include <opencv2/core.hpp>  // Necesario para std::vector<cv::Point>

// Estructura que representa un descriptor de una figura
struct HuDescriptor {
    std::string label;
    std::vector<double> hu;
    std::vector<double> fft;
};

struct ZernikeDescriptor {
    std::string label;
    std::vector<double> zernike;
};


// Funciones de carga y guardado
std::vector<HuDescriptor> cargarCSV(const std::string& ruta);
std::vector<HuDescriptor> cargarCSVDesdeAssets(AAssetManager* mgr, const std::string& nombreArchivo);
bool guardarDescriptorCSV(const std::string& ruta, const HuDescriptor& descriptor, bool append);
std::vector<ZernikeDescriptor> cargarZernikeCSV(const std::string& ruta_csv);
// Cálculo de distancias
double distanciaEuclidea(const std::vector<double>& a, const std::vector<double>& b);
double distanciaCombinada(const std::vector<double>& hu1, const std::vector<double>& fft1,
                          const std::vector<double>& hu2, const std::vector<double>& fft2,
                          double peso_hu = 0.5, double peso_fft = 0.5);

// Cálculo de descriptores
std::vector<double> calcularShapeSignature(std::vector<cv::Point>& contorno);
std::vector<double> calcularFFT(const std::vector<double>& firma);
void completarFFT64(std::vector<double>& fftMag);  // ✅ Función clave para normalizar tamaño del vector FFT

#endif //VISIO_P3_HU_UTILS_H
