// ignore_for_file: prefer_const_constructors, unused_import, unnecessary_import
// тут была Аня
import 'dart:convert';

import 'package:flutter/material.dart';

import 'dart:js_interop';

import 'dart:io';
import 'package:dio/dio.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

import 'package:flutter/cupertino.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';

import 'dart:developer';
import 'package:logger/logger.dart';

void main() {
  runApp(Widget1());
}

class Widget1 extends StatefulWidget {
  const Widget1({Key? key}) : super(key: key);
  @override
  State<Widget1> createState() => MyApp();
}

// ignore: use_key_in_widget_constructors, must_be_immutable
class MyApp extends State<Widget1> {
  List<String> items = [' Скульптура', ' Живопись'];
  String selectedItem = ' Скульптура';
  String textFieldValue = '';
  String resp = "";
  File? _pickedImage;
  Uint8List webImage = Uint8List(8);

  final Logger logger = Logger(); //логи не работают яхз пофиг)))

  String serverText = "Категория"; // текст, полученный от сервера
  String serverDesc = "Здесь будет описание предмета...";
  List<String> serverPhotos = []; // список фотографий, полученных от сервера

  Future<void> _pickImage() async {
    if (!kIsWeb) {
      final ImagePicker _picker = ImagePicker();
      XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        var selected = File(image.path);
        setState(() {
          _pickedImage = selected;
        });

        _sendImageToServer(selected);
      } else {
        logger.d('Изображение не выбрано');
      }
    } else if (kIsWeb) {
      final ImagePicker _picker = ImagePicker();
      XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        var f= await image.readAsBytes();
        setState(() {
          webImage = f;
          _pickedImage = File('a');
        });
        _sendImageToServer(_pickedImage!);
      } else {
        logger.d("Изображение не выбрано");
      }
    } else {
      logger.d("Что-то не так");
    }
  }

  Future<void> _sendImageToServer(File imageFile) async {
    var url = Uri.parse('http://localhost:8000/upload'); //вот сюда ссылку на наш сервер
    var request = http.MultipartRequest('POST', url);
    request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    var response = await request.send();
    resp = response as String;
    if (response.statusCode == 200) {
      log('Изображение успешно загружено');
    } else {
      log('Не удалось загрузить изображение. Ошибка: ${response.reasonPhrase}');
    }
  }

  void _processServerResponse(String response) {
    Map<String, dynamic> responseData = json.decode(response);
    String text = responseData['text'];
    List<String> photos = List<String>.from(responseData['image_requests']);
    setState(() {
      serverText = text;
      serverPhotos = photos;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(
          // backgroundColor: Colors.transparent,
          flexibleSpace: Container(
            alignment: Alignment.bottomLeft,
            width: 100.0,
            height: 100.0,
            color: Colors.white,
            child: Container(
              width: 200.0,
              height: 50.0,
              decoration: BoxDecoration(
                  image:
                      DecorationImage(image: AssetImage('lib/img/logo.png'))),
            ),
          ),
        ),
        body: Container(
            decoration: BoxDecoration(
                image: DecorationImage(
                    image: AssetImage('lib/img/background.png'),
                    fit: BoxFit.cover)),
            margin: const EdgeInsets.symmetric(
                // horizontal: 20.0,
                ),
            child: ListView(scrollDirection: Axis.vertical, children: [
              SizedBox(
                height: 150,
              ),
              Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "Загрузить   ",
                          style: TextStyle(
                              fontSize: 68.0,
                              fontWeight: FontWeight.bold,
                              color: Color.fromRGBO(0, 0, 0, 0.50),
                              fontFamily: "Montserrat"),
                        ),
                        Row(
                          children: [
                            Text(
                              "фото",
                              style: TextStyle(
                                fontSize: 40.0,
                                fontWeight: FontWeight.bold,
                                color: Color.fromRGBO(0, 0, 0, 0.80),
                                fontFamily: "Montserrat",
                              ),
                            ),
                            SizedBox(width: 15),
                            IconButton(
                                onPressed: () {
                                  //загрузка фото
                                  _pickImage();
                                },
                                icon: const Icon(
                                  Icons.download,
                                  color: Colors.white70,
                                ),
                                style: ElevatedButton.styleFrom(
                                    backgroundColor:
                                        Color.fromRGBO(143, 124, 112, 1),
                                    shadowColor: Color(0xff000000))),
                          ],
                        ),
                        SizedBox(
                          height: 30,
                        ),
                        ElevatedButton.icon(
                          onPressed: () {
                            _processServerResponse(resp);
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Color.fromRGBO(143, 124, 112, 1),
                            shadowColor: Color(0xff000000),
                          ),
                          icon: const Icon(
                            Icons.search_sharp,
                            color: Colors.white70,
                          ),
                          label: Text(
                            "Поиск предметов",
                            style: TextStyle(
                              fontSize: 32.0,
                              color: Color(0xffffffff),
                              fontFamily: "Montserrat",
                            ),
                          ),
                        ),
                        // Spacer(flex: 1),
                      ],
                    ),
                    Flexible(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(height: 30),
                          Container(
                            decoration: BoxDecoration(
                              color: Color.fromRGBO(143, 124, 112, 1),
                              borderRadius: BorderRadius.circular(30.0),
                            ),
                          child: Padding(
                            padding: EdgeInsets.all(10.0),
                            child:
                          Text(
                            serverText,
                            style: TextStyle(
                              fontSize: 26.0,
                              color: Colors.white,
                              fontFamily: "Montserrat",
                              ),),
                            ),
                          ),
                          SizedBox(height: 5),
                          SizedBox(
                            width: 130,
                            height: 60,
                            child: Container(
                              alignment: Alignment.centerLeft,
                              decoration: BoxDecoration(
                                  image: DecorationImage(
                                      image: AssetImage('lib/img/cat.png'),
                                      fit: BoxFit.cover)),
                            ),
                          ),
                          Container(
                            width: 350,
                            decoration: BoxDecoration(
                              color: Color.fromRGBO(0, 0, 0, 0.50),
                              borderRadius: BorderRadius.circular(30.0),
                            ),
                            child: Padding(
                              padding: EdgeInsets.all(20.0),
                              child:
                              Text(
                                serverDesc,
                                style: TextStyle(
                                  fontSize: 20.0,
                                  color: Colors.white70,
                                  fontFamily: "Montserrat",
                                ),),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ]),
              Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  SizedBox(height: 40),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: List.generate(
                      5,
                      (index) {
                        int photoIndex = index;
                        return Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 15.0),
                          child: Container(
                            padding: EdgeInsets.all(10.0),
                            width: 160.0,
                            height: 125.0,
                            decoration: BoxDecoration(
                              color: Color.fromRGBO(0, 0, 0, 0.50),
                              borderRadius: BorderRadius.circular(16.0),
                            ),
                            child: serverPhotos.length > photoIndex
                                ? Image.network(serverPhotos[photoIndex])
                                : SizedBox(), // Для пустых фотографий
                          ),
                        );
                      },
                    ),
                  ),
                  SizedBox(height: 20), // Пространство между рядами фотографий
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: List.generate(
                      5,
                      (index) {
                        int photoIndex = index + 5;
                        return Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 15.0),
                          child: Container(
                            padding: EdgeInsets.all(10.0),
                            width: 160.0,
                            height: 125.0,
                            decoration: BoxDecoration(
                              color: Color.fromRGBO(0, 0, 0, 0.50),
                              borderRadius: BorderRadius.circular(16.0),
                            ),
                            child: serverPhotos.length > photoIndex
                                ? Image.network(serverPhotos[photoIndex])
                                : SizedBox(), // Для пустых фотографий
                          ),
                        );
                      },
                    ),
                  ),
                  SizedBox(
                    height: 20,
                  )
                ],
              )

              // Column(
              //   crossAxisAlignment: CrossAxisAlignment.start,
              //   mainAxisAlignment: MainAxisAlignment.start,
              //   children: [
              //     Row(
              //       mainAxisAlignment: MainAxisAlignment.center,
              //       children: [
              //         Container(
              //           padding: EdgeInsets.all(10.0),
              //           width: 160.0,
              //           height: 125.0,
              //           decoration: BoxDecoration(
              //             border:
              //                 Border.all(color: Color.fromRGBO(0, 0, 0, 0.50)),
              //             color: Color.fromRGBO(0, 0, 0, 0.50),
              //             borderRadius: BorderRadius.circular(16.0),
              //           ),
              //         ),
              //         SizedBox(
              //           width: 50,
              //         ),
              //         Container(
              //           padding: EdgeInsets.all(10.0),
              //           width: 160.0,
              //           height: 125.0,
              //           decoration: BoxDecoration(
              //             border:
              //                 Border.all(color: Color.fromRGBO(0, 0, 0, 0.50)),
              //             color: Color.fromRGBO(0, 0, 0, 0.50),
              //             borderRadius: BorderRadius.circular(16.0),
              //           ),
              //         ),
              //         SizedBox(
              //           width: 50,
              //         ),
              //         Container(
              //           padding: EdgeInsets.all(10.0),
              //           width: 160.0,
              //           height: 125.0,
              //           decoration: BoxDecoration(
              //             border:
              //                 Border.all(color: Color.fromRGBO(0, 0, 0, 0.50)),
              //             color: Color.fromRGBO(0, 0, 0, 0.50),
              //             borderRadius: BorderRadius.circular(16.0),
              //           ),
              //         ),
              //         SizedBox(
              //           width: 50,
              //         ),
              //         Container(
              //           padding: EdgeInsets.all(10.0),
              //           width: 160.0,
              //           height: 125.0,
              //           decoration: BoxDecoration(
              //             border:
              //                 Border.all(color: Color.fromRGBO(0, 0, 0, 0.50)),
              //             color: Color.fromRGBO(0, 0, 0, 0.50),
              //             borderRadius: BorderRadius.circular(16.0),
              //           ),
              //         ),
              //         SizedBox(
              //           width: 50,
              //         ),
              //         Container(
              //           padding: EdgeInsets.all(10.0),
              //           width: 160.0,
              //           height: 125.0,
              //           decoration: BoxDecoration(
              //             border:
              //                 Border.all(color: Color.fromRGBO(0, 0, 0, 0.50)),
              //             color: Color.fromRGBO(0, 0, 0, 0.50),
              //             borderRadius: BorderRadius.circular(16.0),
              //           ),
              //         ),
              //       ],
              //     ),
              //     SizedBox(
              //       height: 50,
              //     ),
              //     Row(
              //       mainAxisAlignment: MainAxisAlignment.center,
              //       children: [
              //         Container(
              //           padding: EdgeInsets.all(10.0),
              //           width: 160.0,
              //           height: 125.0,
              //           decoration: BoxDecoration(
              //             border:
              //                 Border.all(color: Color.fromRGBO(0, 0, 0, 0.50)),
              //             color: Color.fromRGBO(0, 0, 0, 0.50),
              //             borderRadius: BorderRadius.circular(16.0),
              //           ),
              //         ),
              //         SizedBox(
              //           width: 50,
              //         ),
              //         Container(
              //           padding: EdgeInsets.all(10.0),
              //           width: 160.0,
              //           height: 125.0,
              //           decoration: BoxDecoration(
              //             border:
              //                 Border.all(color: Color.fromRGBO(0, 0, 0, 0.50)),
              //             color: Color.fromRGBO(0, 0, 0, 0.50),
              //             borderRadius: BorderRadius.circular(16.0),
              //           ),
              //         ),
              //         SizedBox(
              //           width: 50,
              //         ),
              //         Container(
              //           padding: EdgeInsets.all(10.0),
              //           width: 160.0,
              //           height: 125.0,
              //           decoration: BoxDecoration(
              //             border:
              //                 Border.all(color: Color.fromRGBO(0, 0, 0, 0.50)),
              //             color: Color.fromRGBO(0, 0, 0, 0.50),
              //             borderRadius: BorderRadius.circular(16.0),
              //           ),
              //         ),
              //         SizedBox(
              //           width: 50,
              //         ),
              //         Container(
              //           padding: EdgeInsets.all(10.0),
              //           width: 160.0,
              //           height: 125.0,
              //           decoration: BoxDecoration(
              //             border:
              //                 Border.all(color: Color.fromRGBO(0, 0, 0, 0.50)),
              //             color: Color.fromRGBO(0, 0, 0, 0.50),
              //             borderRadius: BorderRadius.circular(16.0),
              //           ),
              //         ),
              //         SizedBox(
              //           width: 50,
              //         ),
              //         Container(
              //           padding: EdgeInsets.all(10.0),
              //           width: 160.0,
              //           height: 125.0,
              //           decoration: BoxDecoration(
              //             border:
              //                 Border.all(color: Color.fromRGBO(0, 0, 0, 0.50)),
              //             color: Color.fromRGBO(0, 0, 0, 0.50),
              //             borderRadius: BorderRadius.circular(16.0),
              //           ),
              //         ),
              //       ],
              //     ),
              //     SizedBox(
              //       height: 50,
              //     )
              //   ],
              // )
            ])),
      ),
    );
  }
}
