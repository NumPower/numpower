--TEST--
NDArray::arcsinh
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::arcsinh($a)->toArray());
print_r(\NDArray::arcsinh($a[0])->toArray());
print_r(\NDArray::arcsinh([[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0.88137358427048
            [1] => 1.4436354637146
        )

    [1] => Array
        (
            [0] => 1.8184465169907
            [1] => 2.0947124958038
        )

)
Array
(
    [0] => 0.88137358427048
    [1] => 1.4436354637146
)
Array
(
    [0] => Array
        (
            [0] => 0.88137358427048
        )

    [1] => Array
        (
            [0] => 1.4436354637146
        )

)