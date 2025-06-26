#include <jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <android/bitmap.h>

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

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_visio_1p3_MainActivity_detectarFigura(JNIEnv *env, jobject thiz, jobject bitmap) {
    // TODO: implement detectarFigura()
    AndroidBitmapInfo info;
    void *pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0)
        return env->NewStringUTF("Error al obtener info");

    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0)
        return env->NewStringUTF("Error al bloquear pixeles");

    cv::Mat img(info.height, info.width, CV_8UC4, pixels);
    cv::Mat gray, binary;
    cv::cvtColor(img, gray, cv::COLOR_RGBA2GRAY);
    cv::threshold(gray, binary, 50, 255, cv::THRESH_BINARY);

    AndroidBitmap_unlockPixels(env, bitmap);

    std::vector<std::vector<cv::Point>> contornos;
    cv::findContours(binary, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contornos.empty())
        return env->NewStringUTF("No se detectaron formas");

    cv::Moments m = cv::moments(contornos[0]);
    double hu[7];
    cv::HuMoments(m, hu);

    return env->NewStringUTF("Figura procesada correctamente ✅");
}