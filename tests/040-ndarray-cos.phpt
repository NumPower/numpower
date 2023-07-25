--TEST--
NDArray::cos
--FILE--
<?php
$a = \NDArray::array([[0, -0.5], [0, -0.5]]);
print_r(\NDArray::cos($a)->toArray());
print_r(\NDArray::cos($a[0])->toArray());
print_r(\NDArray::cos([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 0.87758255004883
        )

    [1] => Array
        (
            [0] => 1
            [1] => 0.87758255004883
        )

)
Array
(
    [0] => 1
    [1] => 0.87758255004883
)
Array
(
    [0] => Array
        (
            [0] => 1
        )

    [1] => Array
        (
            [0] => 0.87758255004883
        )

)