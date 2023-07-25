--TEST--
NDArray::greater
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
$b = \NDArray::array([[5, 6], [7, 8]]);
$c = \NDArray::array([9, 10]);
print_r(\NDArray::greater($a, $b)->toArray());
print_r(\NDArray::greater($a, $a)->toArray());
print_r(\NDArray::greater($c, $c)->toArray());
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