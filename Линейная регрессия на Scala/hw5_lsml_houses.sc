import breeze.linalg.{DenseMatrix, DenseVector, csvwrite, pinv, csvread}
import java.io.File

//Открываем тренировочный фрейм
val df: DenseMatrix[Double] = csvread(new java.io.File("C:\\Games\\housing_data_train.csv"), ',')
val rows: Int = df.rows
val rows_train: Int = rows - 50

//Готовим тренировочную и валидационную части
val x_train: DenseMatrix[Double] = df(0 to rows_train - 1, 0 to 3)

//Довольно сложно оказалось найти вменяемую информацию по тому,
//как делать срезы в DenseMatrix и как по ним итерироваться,
//поэтому местами перескакиваю на массивы
var x_val: Array[Array[Double]] = new Array[Array[Double]](50)
for (i <- rows_train to rows - 1) {
  x_val(i - rows_train) = df(i to i, 0 to 3).toArray
}

val df_train_y = df(0 to rows_train - 1, 13 to 13).toArray
val y_train = DenseVector.zeros[Double](rows_train)
for (i <- 0 to rows_train-1) {
  y_train(i) = df_train_y(i)
}

val df_val_y = df(rows_train to rows - 1, 13 to 13).toArray
val y_val = DenseVector.zeros[Double](50)
for (i <- 0 to 49) {
  y_val(i) = df_val_y(i)
}

//С нуля scala очень тяжело что-то пошла, поэтому самый простой
//алгоритм взял с перемножением матриц, чтобы найти коэффициенты
//перед иксами
val betas: DenseVector[Double] = pinv(x_train.t * x_train) * x_train.t * y_train

var y_pred: Array[Double] = new Array[Double](50)
var rss: Double = 0.0

//Делаем предсказание и считаем среднеквадратичную ошибку
for(i <- 0 to 49) {
  val y = betas(0) * x_val(i)(0) + betas(1) * x_val(i)(1) + betas(2) * x_val(i)(2) + betas(3) * x_val(i)(3)
  y_pred(i) = y
  rss += math.pow((y_val(i) - y), 2)
}

println("MSE=", rss / 50)

val y_pred_out: DenseMatrix[Double] = DenseMatrix(y_pred)

//Записываем ответ на валидационный сет в файл
csvwrite(file=new File("C:\\Games\\housing_data_val_predict.csv"), mat=y_pred_out, separator=',')

//Теперь работаем с тренировочным датасетом

val df_test: DenseMatrix[Double] = csvread(new java.io.File("C:\\Games\\housing_data_test.csv"), ',')
val rows_test: Int = df_test.rows

var x_test: Array[Array[Double]] = new Array[Array[Double]](rows_test)
for (i <- 0 to rows_test - 1) {
  x_test(i) = df(i to i, 0 to 3).toArray
}

var y_pred_test: Array[Double] = new Array[Double](rows_test)
for(i <- 0 to rows_test - 1) {
  val y = betas(0) * x_test(i)(0) + betas(1) * x_test(i)(1) + betas(2) * x_test(i)(2) + betas(3) * x_test(i)(3)
  y_pred_test(i) = y
}

val y_pred_test_out: DenseMatrix[Double] = DenseMatrix(y_pred_test)

//Записываем ответ на тестовый сет в файл
csvwrite(file=new File("C:\\Games\\housing_data_test_predict.csv"), mat=y_pred_test_out, separator=',')
