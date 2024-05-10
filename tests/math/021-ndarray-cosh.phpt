--TEST--
NDArray::cosh
--FILE--
<?php
$a = \NDArray::array([[0, -0.5], [0, -0.5]]);
print_r(\NDArray::cosh($a)->toArray());
print_r(\NDArray::cosh($a[0])->toArray());
print_r(\NDArray::cosh([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 1.1276259422302
        )

    [1] => Array
        (
            [0] => 1
            [1] => 1.1276259422302
        )

)
Array
(
    [0] => 1
    [1] => 1.1276259422302
)
Array
(
    [0] => Array
        (
            [0] => 1
        )

    [1] => Array
        (
            [0] => 1.1276259422302
        )

)