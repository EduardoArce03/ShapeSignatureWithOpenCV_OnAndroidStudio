#include <jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <android/bitmap.h>
#include <android/log.h>
#include "hu_utils.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_visio_1p3_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {

    // Crea una imagen negra 100x100
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);

    // Dibuja un círculo blanco en el centro
    cv::circle(img, cv::Point(50, 50), 25, cv::Scalar(255, 255, 255), -1);

    // Verifica si el píxel central se pintó correctamente
    cv::Vec3b centro = img.at<cv::Vec3b>(50, 50);
    std::string resultado;

    if (centro == cv::Vec3b(255, 255, 255)) {
        resultado = "OpenCV funcionando ✅ - Círculo dibujado.";
    } else {
        resultado = "OpenCV cargado, pero algo falló ❌.";
    }

    return env->NewStringUTF(resultado.c_str());
}


void matToBitmap(JNIEnv* env, const cv::Mat& mat, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    cv::Mat dst(info.height, info.width, CV_8UC4, pixels);

    if (mat.channels() == 1) {
        cv::cvtColor(mat, dst, cv::COLOR_GRAY2RGBA);
    } else if (mat.channels() == 3) {
        cv::cvtColor(mat, dst, cv::COLOR_RGB2RGBA);
    } else {
        mat.copyTo(dst);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_visio_1p3_MainActivity_procesarYMostrar(JNIEnv *env, jobject thiz,
                                                         jobject inputBitmap, jobject outputBitmap) {
    AndroidBitmapInfo info;
    void* pixels;

    AndroidBitmap_getInfo(env, inputBitmap, &info);
    AndroidBitmap_lockPixels(env, inputBitmap, &pixels);
    cv::Mat src(info.height, info.width, CV_8UC4, pixels);

    cv::Mat gray, bin, filled;
    cv::cvtColor(src, gray, cv::COLOR_RGBA2GRAY);
    cv::threshold(gray, bin, 170, 255, cv::THRESH_BINARY);
    AndroidBitmap_unlockPixels(env, inputBitmap);

    std::vector<std::vector<cv::Point>> contornos;
    cv::findContours(bin, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    filled = cv::Mat::zeros(bin.size(), CV_8UC1);
    cv::drawContours(filled, contornos, -1, cv::Scalar(255), cv::FILLED);

    matToBitmap(env, filled, outputBitmap);
}
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_visio_1p3_MainActivity_detectarFigura(JNIEnv *env, jobject thiz, jobject bitmap, jobject assetManager) {
    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0)
        return env->NewStringUTF("Error obteniendo info de bitmap");

    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0)
        return env->NewStringUTF("Error bloqueando píxeles");

    cv::Mat img(info.height, info.width, CV_8UC4, pixels);
    cv::Mat gray, bin, filled;
    cv::cvtColor(img, gray, cv::COLOR_RGBA2GRAY);
    cv::threshold(gray, bin, 170, 255, cv::THRESH_BINARY);
    AndroidBitmap_unlockPixels(env, bitmap);

    std::vector<std::vector<cv::Point>> contornos;
    cv::findContours(bin, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contornos.empty())
        return env->NewStringUTF("No se detectaron figuras");

    filled = cv::Mat::zeros(bin.size(), CV_8UC1);
    cv::drawContours(filled, contornos, -1, cv::Scalar(255), cv::FILLED);

    // Calcular Momentos de Hu
    cv::Moments m = cv::moments(filled, true);
    std::vector<double> hu(7);
    cv::HuMoments(m, hu.data());
    for (double& i : hu)
        i = -1 * copysign(1.0, i) * log10(std::abs(i));

    // Leer CSV desde assets
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    std::vector<HuDescriptor> dataset = cargarCSVDesdeAssets(mgr, "momentos_hu_dataset.csv");

    if (dataset.empty())
        return env->NewStringUTF("Dataset vacío");

    std::string mejorEtiqueta = "Desconocida";
    double mejorDistancia = 1e9;

    for (const auto& entry : dataset) {
        double dist = distanciaEuclidea(hu, entry.hu);
        if (dist < mejorDistancia) {
            mejorDistancia = dist;
            mejorEtiqueta = entry.label;
        }
    }

    std::string resultado = "Figura detectada: " + mejorEtiqueta;
    __android_log_print(ANDROID_LOG_INFO, "NATIVE", "Resultado: %s", resultado.c_str());
    return env->NewStringUTF(resultado.c_str());
}


