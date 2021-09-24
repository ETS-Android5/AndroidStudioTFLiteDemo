package org.techtown.tflitedemo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.io.InputStream;
import java.util.List;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.HexagonDelegate;

import static java.lang.Math.round;

public class MainActivity extends AppCompatActivity {
    //Context context = getApplicationContext();

    private int ipWidth = 224;
    private int ipHeight = 224;
    private int outputSize = 2;
    private boolean grayscale = true;
    private boolean quantized = false;



    private Button getImageButton;
    private Button runButton;
    private Button noDelegateButton;
    private Button nnApiDelegateButton;
    private Button gpuDelegateButton;
    private Button hexagonDelegateButton;
    private Button accCheckButton;
    private ImageView imageView;
    private Bitmap targetImage;
    private TextView textView;
    private TextView textView2;
    private TextView textView3;
    private TextView textView4;

    private Bitmap [] selectedImageList;
    private boolean multiImage = false;

    private String delegateMode;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        checkSelfPermission();

        textView = (TextView) findViewById(R.id.textView);
        textView2 = (TextView) findViewById(R.id.textView2);
        textView3 = (TextView) findViewById(R.id.textView3);
        textView4 = (TextView) findViewById(R.id.textView4);
        imageView = (ImageView) findViewById(R.id.imageView);
        getImageButton = (Button) findViewById(R.id.button2);
        runButton = (Button) findViewById(R.id.button);
        accCheckButton = (Button) findViewById(R.id.button7);
        noDelegateButton = (Button) findViewById(R.id.button5);
        nnApiDelegateButton = (Button) findViewById(R.id.button3);
        gpuDelegateButton = (Button) findViewById(R.id.button4);
        hexagonDelegateButton = (Button) findViewById(R.id.button6);

        noDelegateButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                delegateMode = "CPU";
                textView4.setText(delegateMode);
            }
        });

        nnApiDelegateButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                delegateMode = "NNAPI";
                textView4.setText(delegateMode);
            }
        });

        gpuDelegateButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                delegateMode = "GPU";
                textView4.setText(delegateMode);
            }
        });

        hexagonDelegateButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                delegateMode = "HEXAGON";
                textView4.setText(delegateMode);
            }
        });

        getImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("*/*");
                intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(intent, 1);
            }
        });

        accCheckButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //to check timecost
                long startTimeForReference = 0;
                long endTimeForReference = 1;

                //for delegate Option
                Interpreter.Options options = (new Interpreter.Options());

                NnApiDelegate nnApiDelegate = null;

                GpuDelegate.Options delegateOptions = null;
                GpuDelegate gpuDelegate = null;
                CompatibilityList compatList = new CompatibilityList();

                HexagonDelegate hexagonDelegate = null;


                if(delegateMode == "GPU"){
                    if(compatList.isDelegateSupportedOnThisDevice()){
                        delegateOptions = compatList.getBestOptionsForThisDevice();
                        delegateOptions.setPrecisionLossAllowed(false);
                        delegateOptions.setQuantizedModelsAllowed(true);
                        gpuDelegate = new GpuDelegate(delegateOptions);

                        options.addDelegate(gpuDelegate);
                        //options.setAllowFp16PrecisionForFp32(true);
                        Log.d("delegateMode", "gpudelegate mode");
                    }
                    else{
                        Log.d("delegateMode", "nogpudelegatesupport");
                    }
                }
                else if(delegateMode == "NNAPI"){
                    NnApiDelegate.Options nnapiDelegateOption = new NnApiDelegate.Options();
                    //nnapiDelegateOption.setUseNnapiCpu(true);
                    nnApiDelegate = new NnApiDelegate(nnapiDelegateOption);

                    options.addDelegate(nnApiDelegate);
                    //options.setAllowFp16PrecisionForFp32(true);
                    Log.d("delegateMode", "nnapidelegate mode");
                }
                else if(delegateMode == "HEXAGON"){
                    try{
                        Log.d("debug",MainActivity.this.getApplicationInfo().nativeLibraryDir);
                        hexagonDelegate = new HexagonDelegate(MainActivity.this);
                        options.addDelegate(hexagonDelegate);
                    } catch(UnsupportedOperationException e){
                        Log.d("hexagon not support", "hexagon not supported on device");
                    }
                }
                else{
                    Log.d("fp", "16 allow possible");
                    options.setAllowFp16PrecisionForFp32(false);
                }





                Interpreter tfLDemo = getTfliteInterpreter("Medical_Internal_TFLite_Float16Opt.tflite", options);
                //저장소 폴더 이름
                String path_ben = Environment.getExternalStorageDirectory().getAbsolutePath() + "/Download/internal/0_ben";
                String path_mal = Environment.getExternalStorageDirectory().getAbsolutePath() + "/Download/internal/1_mal";
                Log.d("path", path_ben);

                //먼저 ben_0에 대하여
                File ben_directory = new File(path_ben);
                Log.d("idsdirectory", Boolean.toString(ben_directory.isDirectory()));
                Log.d("check", ben_directory.getAbsolutePath());
                File[] ben_files = ben_directory.listFiles();

                //mal_1에 대해
                File mal_directory = new File(path_mal);
                Log.d("idsdirectory", Boolean.toString(mal_directory.isDirectory()));
                Log.d("check", mal_directory.getAbsolutePath());
                File[] mal_files = mal_directory.listFiles();


                int ben_0_true = 0;
                int mal_1_true = 0;

                for(int i=0; i<ben_files.length; i++){
                    Bitmap target = BitmapFactory.decodeFile(ben_files[i].getAbsolutePath());
                    float[][] outputArray = new float[1][outputSize];

                    //prepare input
                    DataType ipDtype = tfLDemo.getInputTensor(0).dataType();
                    DataType opDtype = tfLDemo.getOutputTensor(0).dataType();
                    //Log.d("Input Data Type", ipDtype.toString());

                    target = Bitmap.createScaledBitmap(target, ipWidth, ipHeight, true);
                    ByteBuffer input;
                    if(quantized){
                        input = grayscaleImageToByteBufferQuantized(target);
                        input.rewind();

                        TensorBuffer probabilityBuffer =
                                TensorBuffer.createFixedSize(new int[]{1, outputSize}, opDtype);
                        //ByteBuffer probabilityBuffer = ByteBuffer.allocateDirect(2);
                        tfLDemo.run(input, probabilityBuffer.getBuffer());
                        //probabilityBuffer.rewind();
                        Log.d("result", ben_files[i].getName() + " : " + Arrays.toString(probabilityBuffer.getIntArray()));
                        if(probabilityBuffer.getIntArray()[0] >= probabilityBuffer.getIntArray()[1]){
                            ben_0_true += 1;
                        }
                    } else {
                        input = grayscaleImageToByteBuffer(target);
                        input.rewind();
                        tfLDemo.run(input, outputArray);

                        Log.d("result", ben_files[i].getName() + " : " + Arrays.toString(outputArray[0]));
                        if(outputArray[0][0] >= outputArray[0][1]){
                            ben_0_true += 1;
                        }
                    }
                }
                Log.d("ben_0_true", Integer.toString(ben_0_true));


                for(int i=0; i<mal_files.length; i++){
                    Bitmap target = BitmapFactory.decodeFile(mal_files[i].getAbsolutePath());
                    float[][] outputArray = new float[1][outputSize];

                    //prepare input
                    DataType ipDtype = tfLDemo.getInputTensor(0).dataType();
                    DataType opDtype = tfLDemo.getOutputTensor(0).dataType();
                    //Log.d("Input Data Type", ipDtype.toString());

                    target = Bitmap.createScaledBitmap(target, ipWidth, ipHeight, true);
                    ByteBuffer input;
                    if(quantized){
                        input = grayscaleImageToByteBufferQuantized(target);
                        input.rewind();
                        TensorBuffer probabilityBuffer =
                                TensorBuffer.createFixedSize(new int[]{1, outputSize}, opDtype);
                        tfLDemo.run(input, probabilityBuffer.getBuffer());
                        Log.d("result", ben_files[i].getName() + " : " + Arrays.toString(probabilityBuffer.getIntArray()));
                        if(probabilityBuffer.getFloatArray()[0] <= probabilityBuffer.getFloatArray()[1]){
                            mal_1_true += 1;
                        }
                    } else {
                        input = grayscaleImageToByteBuffer(target);
                        input.rewind();
                        tfLDemo.run(input, outputArray);
                        Log.d("result", mal_files[i].getName() + " : " + Arrays.toString(outputArray[0]));
                        if(outputArray[0][0] <= outputArray[0][1]){
                            mal_1_true += 1;
                        }
                    }
                }
                Log.d("mal_1_true", Integer.toString(mal_1_true));

            }
        });


        runButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //to check timecost
                long startTimeForReference = 0;
                long endTimeForReference = 1;

                //for delegate Option
                Interpreter.Options options = (new Interpreter.Options());
                //options.setAllowFp16PrecisionForFp32(false);

                NnApiDelegate nnApiDelegate = null;

                GpuDelegate.Options delegateOptions = null;
                GpuDelegate gpuDelegate = null;
                CompatibilityList compatList = new CompatibilityList();

                HexagonDelegate hexagonDelegate = null;


                if(delegateMode == "GPU"){
                    if(compatList.isDelegateSupportedOnThisDevice()){
                        delegateOptions = compatList.getBestOptionsForThisDevice();
                        delegateOptions.setPrecisionLossAllowed(false); //important
                        delegateOptions.setQuantizedModelsAllowed(false);
                        gpuDelegate = new GpuDelegate(delegateOptions);

                        options.addDelegate(gpuDelegate);
                        //options.setAllowFp16PrecisionForFp32(true);
                        Log.d("delegateMode", "gpudelegate mode");
                    }
                    else{
                        Log.d("delegateMode", "nogpudelegatesupport");
                    }
                }
                else if(delegateMode == "NNAPI"){
                    NnApiDelegate.Options nnapiDelegateOption = new NnApiDelegate.Options();
                    //nnapiDelegateOption.setUseNnapiCpu(true);
                    nnApiDelegate = new NnApiDelegate(nnapiDelegateOption);

                    options.addDelegate(nnApiDelegate);
                    //options.setAllowFp16PrecisionForFp32(true);
                    Log.d("delegateMode", "nnapidelegate mode");
                }
                else if(delegateMode == "HEXAGON"){
                    try{
                        Log.d("debug",MainActivity.this.getApplicationInfo().nativeLibraryDir);
                        hexagonDelegate = new HexagonDelegate(MainActivity.this);
                        options.addDelegate(hexagonDelegate);
                    } catch(UnsupportedOperationException e){
                        Log.d("hexagon not support", "hexagon not supported on device");
                    }
                }
                else{
                    Log.d("fp", "16 allow possible");
                    options.setAllowFp16PrecisionForFp32(false);
                }


                if (multiImage) {//batch inference
                    try {
                        Interpreter tfLDemo = getTfliteInterpreter("AlexNet_QAT_INT_uint8io.tflite", options);
                        int[] dimensions = {selectedImageList.length, ipWidth, ipHeight, 3};
                        tfLDemo.resizeInput(0, dimensions);
                        tfLDemo.allocateTensors();

                        //input bytebuffer
                        ByteBuffer inputBatch = ByteBuffer.allocate(selectedImageList.length * ipWidth * ipHeight * 3 * 4); //3 is for the channel, 4 is size of float variable

                        //place for output
                        float[][] outputArray = new float[selectedImageList.length][outputSize];

                        //prepare input
                        DataType ipDtype = tfLDemo.getInputTensor(0).dataType();
                        TensorImage tensorimg = new TensorImage(ipDtype);
                        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                                .add(new NormalizeOp(127.5f, 127.5f))
                                .build();
                        Log.d("batchbuffer capacity: ", Integer.toString(inputBatch.capacity()));
                        Log.d("start", Integer.toString(inputBatch.position()));
                        for(int i=0; i<selectedImageList.length; i++){
                            targetImage = selectedImageList[i];
                            targetImage = Bitmap.createScaledBitmap(targetImage, ipWidth, ipHeight, true);
                            tensorimg.load(targetImage);
                            tensorimg = imageProcessor.process(tensorimg);
                            //Log.d("batchbuffer capacity: ", Integer.toString(tensorimg.getBuffer().capacity()));
                            inputBatch.put(tensorimg.getBuffer());
                        }
                        Log.d("end", Integer.toString(inputBatch.position()));

                        //make inference
                        tfLDemo.run(inputBatch.rewind(), outputArray);
                        startTimeForReference = SystemClock.uptimeMillis();
                        tfLDemo.run(inputBatch.rewind(), outputArray);
                        endTimeForReference = SystemClock.uptimeMillis();
                        tfLDemo.close();
                        if(null != nnApiDelegate){
                            nnApiDelegate.close();
                        }
                        if(null != gpuDelegate){
                            gpuDelegate.close();
                        }
                        if(null != hexagonDelegate){
                            hexagonDelegate.close();
                        }

                        //get index & value
                        int maxIndex = 0;
                        double max = 0;
                        for (int counter = 0; counter < outputArray[1].length; counter++) {
                            if (Double.compare(max, outputArray[1][counter]) < 0) {
                                maxIndex = counter;
                                max = outputArray[1][counter];
                            }
                        }
                        String result = "value: " + Double.toString(max) + ", index: " + Integer.toString(maxIndex);
                        Log.d("alpha", result);
                        textView.setText(result);

                    } catch(Exception e){
                        e.printStackTrace();
                    }
                }
                else{
                    try {
                        /*
                        //Get model

                        options.setAllowFp16PrecisionForFp32(false);
                        Interpreter tfLDemo = getTfliteInterpreter("VGG16_TFLite_Float16Opt.tflite", options);

                        //place for output
                        float[][] outputArray = new float[1][outputSize];

                        //prepare input
                        DataType ipDtype = tfLDemo.getInputTensor(0).dataType();
                        //Log.d("alpha", ipDtype.toString());
                        TensorImage tensorimg = new TensorImage(ipDtype);
                        //Initialize preprocessor
                        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                                .add(new NormalizeOp(127.5f, 127.5f))// 0.0f, 1.0f for quantized model>
                                .build();
                        targetImage = Bitmap.createScaledBitmap(targetImage, ipWidth, ipHeight, true);
                        tensorimg.load(targetImage);
                        tensorimg = imageProcessor.process(tensorimg);
                        /*===============================================================*/
                        //Get model
                        Interpreter tfLDemo = getTfliteInterpreter("Medical_Internal_TFLite_Original.tflite", options);

                        //place for output
                        float[][] outputArray = new float[1][outputSize];

                        //prepare input
                        DataType ipDtype = tfLDemo.getInputTensor(0).dataType();
                        DataType opDtype = tfLDemo.getOutputTensor(0).dataType();
                        Log.d("alpha", ipDtype.toString());
                        TensorImage tensorimg = new TensorImage(ipDtype);//DataType.UINT8
                        //Initialize preprocessor
                        if(grayscale) {
                            targetImage = Bitmap.createScaledBitmap(targetImage, ipWidth, ipHeight, true);
                            ByteBuffer input;
                            if(quantized){
                                input = grayscaleImageToByteBufferQuantized(targetImage);
                            } else {
                                input = grayscaleImageToByteBuffer(targetImage);
                            }
                            TensorBuffer probabilityBuffer =
                                    TensorBuffer.createFixedSize(new int[]{1, outputSize}, opDtype);

                            long[] times = new long[100];
                            input.rewind();
                            tfLDemo.run(input, probabilityBuffer.getBuffer());

                            for (int i = 0; i < 100; i++) {
                                TensorBuffer probabilityBuffer2 =
                                        TensorBuffer.createFixedSize(new int[]{1, outputSize}, opDtype);
                                input.rewind();
                                ByteBuffer p = probabilityBuffer2.getBuffer();
                                startTimeForReference = SystemClock.uptimeMillis();
                                tfLDemo.run(input, p);
                                endTimeForReference = SystemClock.uptimeMillis();

                                times[i] = endTimeForReference - startTimeForReference;
                            }

                            Log.d("result", Arrays.toString(probabilityBuffer.getFloatArray()));
                            Log.d("times", Arrays.toString(times));
                            tfLDemo.close();
                        } else{
                            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                                    .add(new NormalizeOp(0.0f, 1.0f))
                                    .build();
                            targetImage = Bitmap.createScaledBitmap(targetImage, ipWidth, ipHeight, true);
                            tensorimg.load(targetImage);
                            tensorimg = imageProcessor.process(tensorimg);
                            TensorBuffer probabilityBuffer =
                                    TensorBuffer.createFixedSize(new int[]{1, outputSize}, opDtype);
                            //make inference, get times
                            long[] times = new long[100];
                            ByteBuffer b = tensorimg.getBuffer();
                            b.rewind();
                            tfLDemo.run(b, probabilityBuffer.getBuffer());

                            for (int i = 0; i < 100; i++) {
                                TensorBuffer probabilityBuffer2 =
                                        TensorBuffer.createFixedSize(new int[]{1, outputSize}, opDtype);
                                b.rewind();
                                ByteBuffer p = probabilityBuffer2.getBuffer();
                                startTimeForReference = SystemClock.uptimeMillis();
                                tfLDemo.run(b, p);
                                endTimeForReference = SystemClock.uptimeMillis();

                                times[i] = endTimeForReference - startTimeForReference;
                            }
                            Log.d("times", Arrays.toString(times));
                            tfLDemo.close();
                        }
                         /*=================================================*/
                        //make inference, get times
                        /*
                        long[] times = new long[100];
                        tfLDemo.run(tensorimg.getBuffer(), outputArray);
                        /*
                        for(int i=0; i<100; i++) {
                            startTimeForReference = SystemClock.uptimeMillis();
                            tfLDemo.run(tensorimg.getBuffer(), outputArray);
                            endTimeForReference = SystemClock.uptimeMillis();

                            times[i] = endTimeForReference-startTimeForReference;
                        }
                         */
                        /*
                        startTimeForReference = SystemClock.uptimeMillis();
                        tfLDemo.run(tensorimg.getBuffer(), outputArray);
                        endTimeForReference = SystemClock.uptimeMillis();
                        Log.d("times", Arrays.toString(times));
                        tfLDemo.close();
                        */
                        if(null != nnApiDelegate){
                            nnApiDelegate.close();
                        }
                        if(null != gpuDelegate){
                            gpuDelegate.close();
                        }
                        if(null != hexagonDelegate){
                            hexagonDelegate.close();
                        }

                        //get index & value
                        int maxIndex = 0;
                        float max = 0;
                        for (int counter = 0; counter < outputArray[0].length; counter++) {
                            if (Float.compare(max, outputArray[0][counter]) < 0) {
                                maxIndex = counter;
                                max = outputArray[0][counter];
                            }
                        }

                        String result = "value: " + Float.toString(max) + ", index: " + Integer.toString(maxIndex);
                        Log.d("alpha", result);
                        textView.setText(result);

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                String timecost = "Timecost to run model inference: " + (endTimeForReference - startTimeForReference);
                Log.v("beta", timecost);
                textView2.setText(timecost);
            }
        });
    }


    // 이미지 선택 후 다시 돌아왔을 때 ( After selecting imagefile)
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data){
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 1 && resultCode == RESULT_OK){
            if(data.getData() != null) { // 하나의 사진 (Just one picture)
                try {
                    multiImage = false;
                    String selNum = "selected Image: 1";
                    textView3.setText(selNum);
                    Log.v("alpha", data.getData().toString());
                    InputStream is = getContentResolver().openInputStream(data.getData());
                    Bitmap bmSelectedImage = BitmapFactory.decodeStream(is);
                    is.close();
                    imageView.setImageBitmap(bmSelectedImage);
                    targetImage = bmSelectedImage;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            else{
                try{
                    multiImage = true;
                    ClipData clipData = data.getClipData();
                    int selected_photo_num = clipData.getItemCount();
                    selectedImageList = new Bitmap[selected_photo_num];
                    String selNum = "selected Image : " + Integer.toString(selected_photo_num);
                    textView3.setText(selNum);
                    for(int i=0; i<selectedImageList.length; i++){
                        InputStream is = getContentResolver().openInputStream(clipData.getItemAt(i).getUri());
                        Bitmap bmSelectedImage = BitmapFactory.decodeStream(is);
                        is.close();
                        selectedImageList[i] = bmSelectedImage;
                        if(i==0){
                            imageView.setImageBitmap(bmSelectedImage);
                            targetImage = bmSelectedImage;
                        }
                    }
                    Log.i("clipdata number", String.valueOf(selectedImageList.length));
                    Toast.makeText(MainActivity.this, "성공", Toast.LENGTH_LONG).show();
                } catch (Exception e){
                    e.printStackTrace();
                }
            }
        }
        else if(requestCode == 1 && resultCode == RESULT_CANCELED){
            Toast.makeText(this, "취소", Toast.LENGTH_LONG).show();
        }
    }

    // 권한 (permission)
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, int [] grantResults){
        if(requestCode == 1){
            int length = permissions.length;
            for(int i=0; i< length; i++){
                if(grantResults[i] == PackageManager.PERMISSION_GRANTED){
                    Log.d("MainActivity", "권한 허용: "+ permissions[i]);
                }
            }
        }
    }

    public void checkSelfPermission() {
        String temp = "";

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            temp += Manifest.permission.READ_EXTERNAL_STORAGE + " ";
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            temp += Manifest.permission.WRITE_EXTERNAL_STORAGE + " ";
        }

        if (TextUtils.isEmpty(temp) == false) {
            ActivityCompat.requestPermissions(this, temp.trim().split(" "), 1);
        } else {
            Toast.makeText(this, "권한을 모두 허용", Toast.LENGTH_LONG).show();
        }
    }

    // 모델 불러오기 (get TFlite Model)
    private Interpreter getTfliteInterpreter(String modelPath, Interpreter.Options options){
        try{
            return new Interpreter(loadModelFile(MainActivity.this, modelPath), options);
        }
        catch(Exception e){
            e.printStackTrace();
        }
        return null;
    }

    private MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private ByteBuffer grayscaleImageToByteBuffer(Bitmap bitmap){
        ByteBuffer mImgData = ByteBuffer.allocateDirect(4*ipWidth*ipHeight);
        mImgData.order(ByteOrder.nativeOrder());
        mImgData.rewind();
        int [] pixels = new int[ipWidth*ipHeight];
        bitmap.getPixels(pixels, 0, ipWidth, 0, 0, ipWidth, ipHeight);
        for (int pixel : pixels){
            mImgData.putFloat((float) ((0.299*Color.red(pixel) + 0.587*Color.green(pixel) + 0.114*Color.blue(pixel))/255.0));
        }
        mImgData.rewind();
        return mImgData;
    }


    private ByteBuffer grayscaleImageToByteBufferQuantized(Bitmap bitmap){
        ByteBuffer mImgData = ByteBuffer.allocateDirect(ipWidth*ipHeight);
        mImgData.order(ByteOrder.nativeOrder());
        mImgData.rewind();
        int [] pixels = new int[ipWidth*ipHeight];
        bitmap.getPixels(pixels, 0, ipWidth, 0, 0, ipWidth, ipHeight);
        for (int pixel : pixels){
            mImgData.put((byte)(int)round((float)(0.299*Color.red(pixel) + 0.587*Color.green(pixel) + 0.114*Color.blue(pixel))));
        }
        mImgData.rewind();
        return mImgData;
    }
}