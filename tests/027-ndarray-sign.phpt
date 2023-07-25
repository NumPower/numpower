--TEST--
NDArray::sinc
--FILE--
<?php
$a = \NDArray::array([[-156, 150], [19, -39]]);
print_r(\NDArray::sinc($a)->toArray());
print_r(\NDArray::sinc($a[0])->toArray());
print_r(\NDArray::sinc([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => -2.8604226542939E-8
            [1] => -1.3659540165634E-8
        )

    [1] => Array
        (
            [0] => -2.3833271356466E-8
            [1] => 2.8604226542939E-8
        )

)
Array
(
    [0] => -2.8604226542939E-8
    [1] => -1.3659540165634E-8
)
Array
(
    [0] => Array
        (
            [0] => 1
        )

    [1] => Array
        (
            [0] => 0.63661974668503
        )

)