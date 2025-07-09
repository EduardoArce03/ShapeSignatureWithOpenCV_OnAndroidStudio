package com.example.visio_p3;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;

import android.widget.Spinner;
import android.widget.ArrayAdapter;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("opencv_java4");
        System.loadLibrary("native-lib");
    }

    private DrawingView drawingView;
    private ImageView imgResultado;
    private Button btnDetect;
    private TextView txtResultado;
    private Spinner spinDescriptor; // <- nuevo
    private Button btnGuardar;

    // Método nativo nuevo
    public native String clasificarFigura(Bitmap bitmap, String metodo);

    public native String detectarFigura(Bitmap bitmap, android.content.res.AssetManager assetManager);
    public native boolean guardarDescriptorDesdeApp(Bitmap bitmap, String label, String path);
    public native boolean guardarDescriptorZernikeDesdeApp(Bitmap bitmap, String label, String path);

    public native void procesarYMostrar(Bitmap input, Bitmap output);
    public native String predecirDesdeApp(android.content.res.AssetManager assetManager, String metodo);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        txtResultado = findViewById(R.id.txtResultado);
        drawingView = findViewById(R.id.drawingView);
        imgResultado = findViewById(R.id.imgResultado);
        btnDetect = findViewById(R.id.btnDetect);
        btnGuardar = findViewById(R.id.btnGuardar);
        spinDescriptor = findViewById(R.id.spinDescriptor);

// Opciones del Spinner
        String[] opciones = {"Momentos de HU", "Momentos de Zernike"};  // Puedes añadir más

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_item,  // Usa un diseño claro por defecto
                opciones
        );

// Estilo de dropdown
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinDescriptor.setAdapter(adapter);


        btnGuardar.setOnClickListener(v -> {
            Bitmap entrada = drawingView.getBitmap();
            String resultado = detectarFigura(entrada, getAssets());
            String etiqueta = resultado.replace("Figura detectada: ", "").trim();

            String metodo = spinDescriptor.getSelectedItem().toString();
            File archivoTest;

            boolean ok = false;

            if (metodo.equals("Momentos de Zernike")) {
                archivoTest = new File(getFilesDir(), "momentos_zernike_dataset.csv");
                ok = guardarDescriptorZernikeDesdeApp(entrada, etiqueta, archivoTest.getAbsolutePath());
            } else if (metodo.equals("Momentos de HU")) {
                archivoTest = new File(getFilesDir(), "momentos_hu_dataset.csv");
                ok = guardarDescriptorDesdeApp(entrada, etiqueta, archivoTest.getAbsolutePath());
            } else {
                Toast.makeText(this, "Método no válido", Toast.LENGTH_SHORT).show();
                return;
            }

            Toast.makeText(this, ok ? "Guardado exitoso" : "Error al guardar descriptor", Toast.LENGTH_SHORT).show();
        });


        btnDetect.setOnClickListener(v -> {
            Bitmap entrada = drawingView.getBitmap();
            Bitmap salida = Bitmap.createBitmap(
                    entrada.getWidth(), entrada.getHeight(), Bitmap.Config.ARGB_8888);

            procesarYMostrar(entrada, salida);
            imgResultado.setImageBitmap(salida);

            String metodo = spinDescriptor.getSelectedItem().toString(); // ⬅️ selecciona descriptor
            String resultado = clasificarFigura(entrada, metodo);       // usa descriptor

            txtResultado.setText(resultado);
        });

        findViewById(R.id.btnLimpiar).setOnClickListener(v -> drawingView.clear());

        findViewById(R.id.btnEvaluar).setOnClickListener(v -> {
            String metodo = spinDescriptor.getSelectedItem().toString();
            String resultado = predecirDesdeApp(getAssets(), metodo);

            Toast.makeText(this, resultado, Toast.LENGTH_LONG).show();
        });
    }

    public native String stringFromJNI();
}
