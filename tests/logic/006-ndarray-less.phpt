--TEST--
NDArray::less
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
$b = \NDArray::array([[5, 6], [7, 8]]);
$c = \NDArray::array([9, 10]);
print_r(\NDArray::less($a, $b)->toArray());
print_r(\NDArray::less($a, $a)->toArray());
print_r(\NDArray::less($c, $c)->toArray());
print_r(\NDArray::less($a, $c)->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 1
        )

    [1] => Array
        (
            [0] => 1
            [1] => 1
        )

)
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 0
        )

    [1] => Array
        (
            [0] => 0
            [1] => 0
        )

)
Array
(
    [0] => 0
    [1] => 0
)

Fatal error: Uncaught Error: Incompatible shapes in `equal` function in /src/tests/logic/006-ndarray-less.php:8
Stack trace:
#0 /src/tests/logic/006-ndarray-less.php(8): NDArray::less(Object(NDArray), Object(NDArray))
#1 {main}
  thrown in /src/tests/logic/006-ndarray-less.php on line 8