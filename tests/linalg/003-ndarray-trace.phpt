--TEST--
NDArray::trace
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
$b = \NDArray::array([[5, 6], [7, 8]]);
$c = \NDArray::array([9, 10]);
$d = \NDArray::array([[9], [10]]);
print_r(\NDArray::trace($a));
print_r(\NDArray::trace([[1, 2], [3, 4]]));
print_r(\NDArray::trace($d));
print_r(\NDArray::trace($c));
?>
--EXPECT--
559
Fatal error: Uncaught Error: NDArray_Diagonal: Array must be 2-d. in /src/tests/linalg/003-ndarray-trace.php:9
Stack trace:
#0 /src/tests/linalg/003-ndarray-trace.php(9): NDArray::trace(Object(NDArray))
#1 {main}
  thrown in /src/tests/linalg/003-ndarray-trace.php on line 9