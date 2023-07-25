--TEST--
NDArray::greater_equal
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
$b = \NDArray::array([[5, 6], [7, 8]]);
$c = \NDArray::array([9, 10]);
print_r(\NDArray::greater_equal($a, $b)->toArray());
print_r(\NDArray::greater_equal($a, $a)->toArray());
print_r(\NDArray::greater_equal($c, $c)->toArray());
print_r(\NDArray::greater_equal($a, $c)->toArray());
?>
--EXPECT--
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
    [0] => 1
    [1] => 1
)

Fatal error: Uncaught Error: Incompatible shapes in `equal` function in /src/tests/logic/005-ndarray-greater_equal.php:8
Stack trace:
#0 /src/tests/logic/005-ndarray-greater_equal.php(8): NDArray::greater_equal(Object(NDArray), Object(NDArray))
#1 {main}
  thrown in /src/tests/logic/005-ndarray-greater_equal.php on line 8