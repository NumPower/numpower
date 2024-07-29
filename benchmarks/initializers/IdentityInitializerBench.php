<?php
    /**
     * @Groups({"initializers"})
     */
    class IdentityInitializerBench
    {
        /**
        * @var size
        */
        private $size = 0;

        public function setUp(array $params): void
        {
            $this->size = $params['size'];
        }
        
        /**
        * @BeforeMethods("setUp")
        * @Revs(1000)
        * @Iterations(5)
        * @ParamProviders({
        *     "provideSizes",
        * })
        */
        public function benchIdentity($params)
        {
            $ndarray = \NDArray::identity($this->size);
        }

        public function provideSizes() {
            yield ['size' => 100];
            yield ['size' => 500];
            yield ['size' => 1000];
        }
    }
?>