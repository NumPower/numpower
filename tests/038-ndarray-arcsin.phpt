--TEST--
NDArray::arcsin
--FILE--
<?php
$a = \NDArray::array([[0, -0.5], [0, -0.5]]);
print_r(\NDArray::arcsin($a)->toArray());
print_r(\NDArray::arcsin($a[0])->toArray());
print_r(\NDArray::arcsin([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => -0.52359879016876
        )

    [1] => Array
        (
            [0] => 0
            [1] => -0.52359879016876
        )

)
Array
(
    [0] => 0
    [1] => -0.52359879016876
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => -0.52359879016876
        )

)