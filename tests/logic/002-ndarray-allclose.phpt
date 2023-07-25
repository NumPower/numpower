--TEST--
NDArray::allclose
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
$b = \NDArray::array([[5, 6], [7, 8]]);
$c = \NDArray::array([9, 10]);
print_r(\NDArray::allclose($a, $b));
print_r(\NDArray::allclose($a, $a));
print_r(\NDArray::allclose($c, $c));
print_r(\NDArray::allclose($a, $c));
?>
--EXPECT--
11
Fatal error: Uncaught Error: Shape mismatch in /src/tests/logic/002-ndarray-allclose.php:8
Stack trace:
#0 /src/tests/logic/002-ndarray-allclose.php(8): NDArray::allclose(Object(NDArray), Object(NDArray))
#1 {main}
  thrown in /src/tests/logic/002-ndarray-allclose.php on line 8