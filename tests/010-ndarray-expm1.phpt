--TEST--
NDArray::expm1
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::expm1($a)->toArray());
print_r(\NDArray::expm1($a[0])->toArray());
print_r(\NDArray::expm1([[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 1.7182817459106
            [1] => 6.3890562057495
        )

    [1] => Array
        (
            [0] => 19.085536956787
            [1] => 53.598152160645
        )

)
Array
(
    [0] => 1.7182817459106
    [1] => 6.3890562057495
)
Array
(
    [0] => Array
        (
            [0] => 1.7182817459106
        )

    [1] => Array
        (
            [0] => 6.3890562057495
        )

)