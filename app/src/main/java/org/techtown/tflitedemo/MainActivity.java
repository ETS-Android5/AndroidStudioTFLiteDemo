package org.techtown.tflitedemo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.io.InputStream;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class MainActivity extends AppCompatActivity {
    private int ipWidth = 224;
    private int ipHeight = 224;
    private int outputSize = 1001;

    private Button getImageButton;
    private Button runButton;
    private ImageView imageView;
    private Bitmap targetImage;
    private TextView textView;
    private TextView textView2;
    private TextView textView3;

    private Bitmap [] selectedImageList;
    private boolean multiImage = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        checkSelfPermission();

        textView = (TextView) findViewById(R.id.textView);
        textView2 = (TextView) findViewById(R.id.textView2);
        textView3 = (TextView) findViewById(R.id.textView3);
        imageView = (ImageView) findViewById(R.id.imageView);
        getImageButton = (Button) findViewById(R.id.button2);
        runButton = (Button) findViewById(R.id.button);

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

        runButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //to check timecost
                long startTimeForReference = 0;
                long endTimeForReference = 1;
                if (multiImage) {//batch inference
                    try {
                        //Get model
                        Interpreter tfLDemo = getTfliteInterpreter("mobilenet_v1_1.0_224.tflite");
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
                        startTimeForReference = SystemClock.uptimeMillis();
                        tfLDemo.run(inputBatch.rewind(), outputArray);
                        endTimeForReference = SystemClock.uptimeMillis();
                        tfLDemo.close();

                        //get index & value
                        int maxIndex = 0;
                        float max = 0;
                        for (int counter = 0; counter < outputArray[1].length; counter++) {
                            if (Float.compare(max, outputArray[1][counter]) < 0) {
                                maxIndex = counter;
                                max = outputArray[1][counter];
                            }
                        }
                        String result = "value: " + Float.toString(max) + ", index: " + Integer.toString(maxIndex);
                        Log.d("alpha", result);
                        textView.setText(result);

                    } catch(Exception e){
                        e.printStackTrace();
                    }
                }
                else{
                    try {
                        //Get model
                        Interpreter tfLDemo = getTfliteInterpreter("mobilenet_v1_1.0_224.tflite");

                        //place for output
                        float[][] outputArray = new float[1][outputSize];

                        //prepare input
                        DataType ipDtype = tfLDemo.getInputTensor(0).dataType();
                        //Log.d("alpha", ipDtype.toString());
                        TensorImage tensorimg = new TensorImage(ipDtype);
                        //Initialize preprocessor
                        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                                .add(new NormalizeOp(127.5f, 127.5f))
                                .build();
                        targetImage = Bitmap.createScaledBitmap(targetImage, ipWidth, ipHeight, true);
                        tensorimg.load(targetImage);
                        tensorimg = imageProcessor.process(tensorimg);

                        //make inference
                        startTimeForReference = SystemClock.uptimeMillis();
                        tfLDemo.run(tensorimg.getBuffer(), outputArray);
                        endTimeForReference = SystemClock.uptimeMillis();
                        tfLDemo.close();

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
    private Interpreter getTfliteInterpreter(String modelPath){
        try{
            return new Interpreter(loadModelFile(MainActivity.this, modelPath));
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


}